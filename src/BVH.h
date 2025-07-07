#pragma once
#include "HittableList.h"
#include "Ray.h"

#define SHARED_STACK_SIZE 8
#define LOCAL_STACK_SIZE  8
#define STACK_SIZE		  (SHARED_STACK_SIZE + LOCAL_STACK_SIZE)
#define MAX_THREADS		  64 // Match your block size

namespace BVH
{
	// Note: Could use this as a class with member functions but NVCC wouldn't show them in PTXAS info
	struct alignas(32) BVH
	{
		struct BVHNode
		{
			uint32_t Left;	// Index of left child (or sphere index for leaves)
			uint32_t Right; // Index of right child (unused for leaves)
		};
		BVHNode* __restrict__ m_Nodes;
		AABB* __restrict__ m_Bounds;

		uint32_t m_Count = 0;
		uint32_t m_Root	 = 0;
	};

	template<ExecutionMode Mode>
	[[nodiscard]] __host__ inline BVH* Init(const uint32_t capacity)
	{
		BVH h_BVH;
		h_BVH.m_Bounds = MemPolicy<Mode>::template Alloc<AABB>(capacity);
		h_BVH.m_Nodes  = MemPolicy<Mode>::template Alloc<BVH::BVHNode>(capacity);

		BVH* d_BVH = MemPolicy<Mode>::template Alloc<BVH>(1);

		if constexpr (Mode == ExecutionMode::GPU)
			CHECK_CUDA_ERRORS(cudaMemcpy(d_BVH, &h_BVH, sizeof(BVH), cudaMemcpyHostToDevice));
		else
			*d_BVH = h_BVH;

		return d_BVH;
	}

	template<ExecutionMode Mode>
	__host__ inline void Destroy(BVH* bvh)
	{
		if constexpr (Mode == ExecutionMode::GPU)
		{
			BVH hostBVH;
			CHECK_CUDA_ERRORS(cudaMemcpy(&hostBVH, bvh, sizeof(BVH), cudaMemcpyDeviceToHost));

			MemPolicy<Mode>::Free(hostBVH.m_Nodes);
			MemPolicy<Mode>::Free(hostBVH.m_Bounds);
			MemPolicy<Mode>::Free(bvh);
		}
		else
		{
			MemPolicy<Mode>::Free(bvh->m_Nodes);
			MemPolicy<Mode>::Free(bvh->m_Bounds);
			MemPolicy<Mode>::Free(bvh);
		}
	}

	[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE uint32_t AddNode(const uint32_t leftIdx, const uint32_t rightIdx, const AABB& box)
	{
		const RenderParams* __restrict__ params		= GetParams();
		params->BVH->m_Nodes[params->BVH->m_Count]	= { .Left = leftIdx, .Right = rightIdx };
		params->BVH->m_Bounds[params->BVH->m_Count] = box;

		return params->BVH->m_Count++;
	}

#ifdef RTIOW_BVH_VEB // can be more significant if we have a lot of nodes that they don't fit L1 cache
	__device__ __host__ void ReorderVEB(uint32_t nodeIdx, uint32_t* nodeMap, BVH::BVHNode* tempNodes, AABB* tempBounds, uint32_t& newIndex, int currentDepth, int treeHeight);

	// Helper function for recursion with depth limits
	__device__ __host__ CPU_ONLY_INLINE void ReorderVEBRecursive(uint32_t nodeIdx, uint32_t* nodeMap, BVH::BVHNode* tempNodes, AABB* tempBounds, uint32_t& newIndex, int currentDepth, int depthLimit, int treeHeight)
	{
		// Return if node is invalid
		if (nodeIdx == UINT32_MAX)
			return;

		const RenderParams* __restrict__ params = GetParams();

		// Get the current node
		const BVH::BVHNode& node = params->BVH->m_Nodes[nodeIdx];

		// For leaf nodes, just add them to the new array
		if (node.Right == UINT32_MAX)
		{
			// Store the mapping from old to new index
			nodeMap[nodeIdx] = newIndex;

			// Copy the node and bounds to their new positions
			tempNodes[newIndex]	 = node;
			tempBounds[newIndex] = params->BVH->m_Bounds[nodeIdx];

			// Move to next index
			newIndex++;
			return;
		}

		// If we're at the depth limit, process this subtree using VEB
		if (currentDepth >= depthLimit)
		{
			ReorderVEB(nodeIdx, nodeMap, tempNodes, tempBounds, newIndex, currentDepth, treeHeight);
			return;
		}

		// Otherwise, first add the node itself
		nodeMap[nodeIdx]	 = newIndex;
		tempNodes[newIndex]	 = node;
		tempBounds[newIndex] = params->BVH->m_Bounds[nodeIdx];
		newIndex++;

		// Then recurse on left and right children
		ReorderVEBRecursive(node.Left, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, depthLimit, treeHeight);
		ReorderVEBRecursive(node.Right, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, depthLimit, treeHeight);
	}

	// Recursive function to reorder the nodes in a van Emde Boas layout
	__device__ __host__ CPU_ONLY_INLINE void ReorderVEB(uint32_t nodeIdx, uint32_t* nodeMap, BVH::BVHNode* tempNodes, AABB* tempBounds, uint32_t& newIndex, int currentDepth, int treeHeight)
	{
		// Return if node is invalid
		if (nodeIdx == UINT32_MAX)
			return;
		const RenderParams* __restrict__ params = GetParams();

		// Get the current node
		const BVH::BVHNode& node = params->BVH->m_Nodes[nodeIdx];

		// For leaf nodes, just add them to the new array
		if (node.Right == UINT32_MAX)
		{
			// Store the mapping from old to new index
			nodeMap[nodeIdx] = newIndex;

			// Copy the node and bounds to their new positions
			tempNodes[newIndex]	 = node;
			tempBounds[newIndex] = params->BVH->m_Bounds[nodeIdx];

			// Move to next index
			newIndex++;
			return;
		}

		// Calculate the height of the remaining subtree
		int height		= treeHeight - currentDepth;
		int upperHeight = height / 2;

		// For internal nodes, first add the node itself
		nodeMap[nodeIdx]	 = newIndex;
		tempNodes[newIndex]	 = node;
		tempBounds[newIndex] = params->BVH->m_Bounds[nodeIdx];
		newIndex++;

		// Then recurse on top part of tree (upper levels)
		ReorderVEBRecursive(node.Left, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, currentDepth + upperHeight, treeHeight);

		// Then recurse on bottom part of tree (lower levels)
		ReorderVEBRecursive(node.Right, nodeMap, tempNodes, tempBounds, newIndex, currentDepth + 1, currentDepth + upperHeight, treeHeight);
	}

	__device__ __host__ CPU_ONLY_INLINE uint32_t ReorderBVH()
	{
		const RenderParams* __restrict__ params = GetParams();

		// Temporary arrays to hold the reordered BVH
		BVH::BVHNode* tempNodes	 = static_cast<BVH::BVHNode*>(malloc(params->BVH->m_Count * sizeof(BVH::BVHNode)));
		AABB*		  tempBounds = static_cast<AABB*>(malloc(params->BVH->m_Count * sizeof(AABB)));

		// Create a map to store the old -> new node indices mapping
		uint32_t* nodeMap = static_cast<uint32_t*>(malloc(params->BVH->m_Count * sizeof(uint32_t)));

		// Calculate the height of the tree (approximate for potentially unbalanced trees)
		// For a binary tree with n nodes, height is approximately log2(n)
		int treeHeight = static_cast<int>(log2f(static_cast<float>(params->BVH->m_Count))) + 1;

		// First pass: Copy nodes to temp arrays in van Emde Boas layout
		uint32_t newIndex = 0;
		ReorderVEB(params->BVH->m_Root, nodeMap, tempNodes, tempBounds, newIndex, 0, treeHeight);

		// Save the new root index
		uint32_t newRootIndex = nodeMap[params->BVH->m_Root];

		// Second pass: Update child indices using the mapping
		for (uint32_t i = 0; i < params->BVH->m_Count; ++i)
		{
			// If not a leaf node, update the child indices
			if (tempNodes[i].Right != UINT32_MAX)
			{
				tempNodes[i].Left  = nodeMap[tempNodes[i].Left];
				tempNodes[i].Right = nodeMap[tempNodes[i].Right];
			}
		}

		// Copy back the reordered data
		memcpy(params->BVH->m_Nodes, tempNodes, params->BVH->m_Count * sizeof(BVH::BVHNode));
		memcpy(params->BVH->m_Bounds, tempBounds, params->BVH->m_Count * sizeof(AABB));

		// Clean up
		free(tempNodes);
		free(tempBounds);
		free(nodeMap);

		// Return the new root index
		return newRootIndex;
	}
#endif

	[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE uint32_t Build(uint32_t* indices, uint32_t start, uint32_t end)
	{
		const RenderParams* __restrict__ params = GetParams();
		const uint32_t objectSpan				= end - start;

		// Compute bounding box for this node
		AABB box;
		bool firstBox = true;
		for (uint32_t i = start; i < end; ++i)
		{
			uint32_t	sphereIndex = indices[i];
			const AABB& currentBox	= params->List->AABBs[sphereIndex];
			if (firstBox)
			{
				box		 = currentBox;
				firstBox = false;
			}
			else
			{
				box = AABB(box, currentBox);
			}
		}

		// Handle leaf cases
		if (objectSpan == 1)
		{
			// Leaf node: store sphere index
			return AddNode(indices[start], UINT32_MAX, box);
		}

		if (objectSpan == 2)
		{
			uint32_t idxA = indices[start];
			uint32_t idxB = indices[start + 1];

			// Create leaf nodes for each sphere
			const AABB& boxA = params->List->AABBs[idxA];
			const AABB& boxB = params->List->AABBs[idxB];

			uint32_t leftLeaf  = AddNode(idxA, UINT32_MAX, boxA);
			uint32_t rightLeaf = AddNode(idxB, UINT32_MAX, boxB);

			// Create parent internal node
			const AABB combined(boxA, boxB);
			return AddNode(leftLeaf, rightLeaf, combined);
		}

		// Use SAH to find the best split
		float	 bestCost	  = FLT_MAX;
		uint32_t bestAxis	  = 0;
		uint32_t bestSplitIdx = start + objectSpan / 2; // Default middle split as fallback

		// Cost of not splitting (creating a leaf)
		// float no_split_cost = objectSpan * box.SurfaceArea();

		// Try each axis
		for (uint32_t axis = 0; axis < 3; ++axis)
		{
			// Sort indices along this axis
			auto comparator = [&](const uint32_t a, const uint32_t b)
			{
				return params->List->AABBs[a].Center()[axis] < params->List->AABBs[b].Center()[axis];
			};

			thrust::sort(indices + start, indices + end, comparator);

			// Precompute all bounding boxes from left to right
			AABB* leftBoxes = (AABB*)malloc(sizeof(AABB) * objectSpan);
			leftBoxes[0]	= params->List->AABBs[indices[start]];
			for (uint32_t i = 1; i < objectSpan; ++i)
			{
				leftBoxes[i] = AABB(leftBoxes[i - 1], params->List->AABBs[indices[start + i]]);
			}

			// Precompute all bounding boxes from right to left
			AABB* rightBoxes		   = (AABB*)malloc(sizeof(AABB) * objectSpan);
			rightBoxes[objectSpan - 1] = params->List->AABBs[indices[end - 1]];
			for (int i = objectSpan - 2; i >= 0; --i)
			{
				rightBoxes[i] = AABB(rightBoxes[i + 1], params->List->AABBs[indices[start + i]]);
			}

			// Evaluate SAH cost for each possible split
			for (uint32_t i = 1; i < objectSpan; ++i)
			{
				const float leftCount  = i;
				const float rightCount = (float)(objectSpan - i);

				const float leftSa	= leftBoxes[i - 1].SurfaceArea();
				const float rightSa = rightBoxes[i].SurfaceArea();

				// SAH cost formula: C = T_traverse + (left_count * left_sa + right_count * right_sa) / parent_sa * T_intersect
				// We can simplify by using constant traversal and intersection costs
				constexpr float traversalCost	 = 0.3f;
				constexpr float intersectionCost = 1.0f;
				const float		parentSA		 = box.SurfaceArea();
				const float		cost			 = traversalCost + (leftCount * leftSa + rightCount * rightSa) / parentSA * intersectionCost;
				if (cost < bestCost)
				{
					bestCost	 = cost;
					bestAxis	 = axis;
					bestSplitIdx = start + i;
				}
			}

			free(leftBoxes);
			free(rightBoxes);
		}

		// If no split is better than not splitting, and we have fewer than some threshold of objects,
		// we could make this a leaf. However, we'll always split for simplicity and compatibility
		// with the traversal function.

		// Resort along the best axis if it's not the last one we tried
		if (bestAxis != 2)
		{
			auto comparator = [&](const uint32_t a, uint32_t b)
			{
				return params->List->AABBs[a].Center()[bestAxis] < params->List->AABBs[b].Center()[bestAxis];
			};

			thrust::sort(indices + start, indices + end, comparator);
		}

		// Recursively build children
		const uint32_t leftIdx	= Build(indices, start, bestSplitIdx);
		const uint32_t rightIdx = Build(indices, bestSplitIdx, end);

		// Compute the combined bounding box from the actual child nodes
		AABB combined = params->BVH->m_Bounds[leftIdx];

		const AABB& rightAabb = params->BVH->m_Bounds[rightIdx];
		combined			  = AABB(combined, rightAabb);

		return AddNode(leftIdx, rightIdx, combined);
	}

	inline __device__ void StackPush(uint32_t value, int& stackPtr, uint16_t* sharedStackData, uint16_t* localStackData)
	{
		if (stackPtr < SHARED_STACK_SIZE)
			sharedStackData[stackPtr++] = value;
		else
			localStackData[stackPtr++ - SHARED_STACK_SIZE] = value;
	}

	inline __device__ uint16_t StackPop(int& stackPtr, const uint16_t* sharedStackData, const uint16_t* localStackData)
	{
		if (stackPtr <= SHARED_STACK_SIZE)
			return sharedStackData[--stackPtr];
		else
			return localStackData[--stackPtr - SHARED_STACK_SIZE];
	}

	[[nodiscard]] __device__ __host__ CPU_ONLY_INLINE bool Traverse(const Ray& ray, const float tmin, float tmax, HitRecord& bestHit)
	{
		const RenderParams* __restrict__ params = GetParams();

		// Use registers for stack instead of memory
		uint16_t currentNode = params->BVH->m_Root;

#ifdef __CUDA_ARCH__
		__shared__ uint16_t sharedStack[(MAX_THREADS * SHARED_STACK_SIZE * sizeof(uint16_t) + 3) / 4];
#	define THREAD_INDEX (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
#	define DECLARE_STACK()                                                         \
		uint16_t* sharedStackData = &sharedStack[THREAD_INDEX * SHARED_STACK_SIZE]; \
		uint16_t  localStackData[LOCAL_STACK_SIZE]
#else
#	define DECLARE_STACK() uint16_t stackData[STACK_SIZE]
#endif

		DECLARE_STACK();
		int	 stackPtr	 = 0;
		bool hitAnything = false;

		// Pre-compute ray inverse direction once
		const Vec3 invDir				 = 1.0f / ray.Direction;
		const Vec3 rayOriginMulNegInvDir = -ray.Origin * invDir;

#ifdef __CUDA_ARCH__
		StackPush(currentNode, stackPtr, sharedStackData, localStackData);
#else
		stackData[stackPtr++] = currentNode;
#endif

		// Front-to-back traversal for early termination
		while (stackPtr != 0) [[likely]]
		{
			assert(stackPtr < 16);

			const BVH::BVHNode& node = params->BVH->m_Nodes[currentNode];

			// Process leaf node
			if (node.Right == UINT32_MAX) [[unlikely]]
			{
				// Process hit test
				hitAnything |= Hitables::IntersectPrimitive(ray, tmin, tmax, bestHit, node.Left);

#ifdef __CUDA_ARCH__
				currentNode = StackPop(stackPtr, sharedStackData, localStackData);

#else
				currentNode = stackData[--stackPtr];
#endif
				continue;
			}

#ifdef __CUDA_ARCH__

			// Check left child
			const AABB& leftBounds = params->BVH->m_Bounds[node.Left];
			float		tx0 = std::fmaf(invDir.x, leftBounds.Min.x, rayOriginMulNegInvDir.x), tx1 = std::fmaf(invDir.x, leftBounds.Max.x, rayOriginMulNegInvDir.x);
			float		ty0 = std::fmaf(invDir.y, leftBounds.Min.y, rayOriginMulNegInvDir.y), ty1 = std::fmaf(invDir.y, leftBounds.Max.y, rayOriginMulNegInvDir.y);
			float		tz0 = std::fmaf(invDir.z, leftBounds.Min.z, rayOriginMulNegInvDir.z), tz1 = std::fmaf(invDir.z, leftBounds.Max.z, rayOriginMulNegInvDir.z);
			float		tEnterL = std::fmaxf(std::fmaxf(std::fminf(tx0, tx1), std::fminf(ty0, ty1)), std::fmaxf(std::fminf(tz0, tz1), tmin));
			float		tExitL	= std::fminf(std::fminf(std::fmaxf(tx0, tx1), std::fmaxf(ty0, ty1)), std::fminf(std::fmaxf(tz0, tz1), tmax));
			bool		hitLeft = tEnterL <= tExitL;

			// Check right child
			const AABB& rightBounds = params->BVH->m_Bounds[node.Right];
			tx0 = std::fmaf(invDir.x, rightBounds.Min.x, rayOriginMulNegInvDir.x), tx1 = std::fmaf(invDir.x, rightBounds.Max.x, rayOriginMulNegInvDir.x);
			ty0 = std::fmaf(invDir.y, rightBounds.Min.y, rayOriginMulNegInvDir.y), ty1 = std::fmaf(invDir.y, rightBounds.Max.y, rayOriginMulNegInvDir.y);
			tz0 = std::fmaf(invDir.z, rightBounds.Min.z, rayOriginMulNegInvDir.z), tz1 = std::fmaf(invDir.z, rightBounds.Max.z, rayOriginMulNegInvDir.z);
			const float tEnterR	 = std::fmaxf(std::fmaxf(std::fminf(tx0, tx1), std::fminf(ty0, ty1)), std::fmaxf(std::fminf(tz0, tz1), tmin));
			const float tExitR	 = std::fminf(std::fminf(std::fmaxf(tx0, tx1), std::fmaxf(ty0, ty1)), std::fminf(std::fmaxf(tz0, tz1), tmax));
			bool		hitRight = tEnterR <= tExitR;

			uint32_t closerChild  = tEnterL > tEnterR ? node.Right : node.Left;
			uint32_t fartherChild = tEnterL > tEnterR ? node.Left : node.Right;

#else
			float tEnterL = 0.0f, tEnterR = 0.0f;

			auto SlabTest = [&](const AABB& b, float& tEnterOut) -> bool
			{
				float tx0	 = std::fma(invDir.x, b.Min.x, rayOriginMulNegInvDir.x);
				float tx1	 = std::fma(invDir.x, b.Max.x, rayOriginMulNegInvDir.x);
				float tEnter = glm::max(glm::min(tx0, tx1), tmin);
				float tExit	 = glm::min(glm::max(tx0, tx1), tmax);
				if (tExit < tEnter)
					return false;

				float ty0 = std::fma(invDir.y, b.Min.y, rayOriginMulNegInvDir.y);
				float ty1 = std::fma(invDir.y, b.Max.y, rayOriginMulNegInvDir.y);
				tEnter	  = glm::max(glm::min(ty0, ty1), tEnter);
				tExit	  = glm::min(glm::max(ty0, ty1), tExit);
				if (tExit < tEnter)
					return false;

				float tz0 = std::fma(invDir.z, b.Min.z, rayOriginMulNegInvDir.z);
				float tz1 = std::fma(invDir.z, b.Max.z, rayOriginMulNegInvDir.z);
				tEnter	  = glm::max(glm::min(tz0, tz1), tEnter);
				tExit	  = glm::min(glm::max(tz0, tz1), tExit);
				if (tExit < tEnter)
					return false;

				tEnterOut = tEnter;
				return true;
			};

			const bool hitLeft	= SlabTest(params->BVH->m_Bounds[node.Left], tEnterL);
			const bool hitRight = SlabTest(params->BVH->m_Bounds[node.Right], tEnterR);

			uint32_t closerChild  = tEnterL > tEnterR ? node.Right : node.Left;
			uint32_t fartherChild = tEnterL > tEnterR ? node.Left : node.Right;

#endif

			// Resolve everything immediately
			if (hitLeft && hitRight)
			{
				// Resolve early which side is first
				currentNode = closerChild;
#ifdef __CUDA_ARCH__
				StackPush(fartherChild, stackPtr, sharedStackData, localStackData);
#else
				stackData[stackPtr++] = fartherChild;
#endif
			}
			else if (hitLeft ^ hitRight)
			{
				currentNode = hitLeft ? node.Left : node.Right;
			}
			else // noneHit
			{
				if (stackPtr == 0)
					break;
#ifdef __CUDA_ARCH__
				currentNode = StackPop(stackPtr, sharedStackData, localStackData);

#else
				currentNode = stackData[--stackPtr];
#endif
			}
		}
		return hitAnything;
	}

} // namespace BVH
