#include "BVH.h"
#include "Sphere.h"
#include <EASTL/vector.h>


void* __cdecl operator new[](size_t size, char const*, int, unsigned int, char const*, int)
{
	return malloc(size);
}

void * __cdecl operator new[](unsigned __int64 size,unsigned __int64,unsigned __int64,char const *,int,unsigned int,char const *,int)
{
	return malloc(size);
}

inline bool box_compare(const Hittable* a, const Hittable* b, int axis) {
	AABB box_a;
	AABB box_b;

	if (!a->BoundingBox(0, 0, box_a) || !b->BoundingBox(0, 0, box_b))
		LOG_CORE_ERROR("No bounding box in bvh_node constructor.");

	return box_a.Min()[axis] < box_b.Min()[axis];
}


bool box_x_compare(const Hittable* a, const Hittable* b) {
	return box_compare(a, b, 0);
}

bool box_y_compare(const Hittable* a, const Hittable* b) {
	return box_compare(a, b, 1);
}

bool box_z_compare(const Hittable* a, const Hittable* b) {
	return box_compare(a, b, 2);
}



BVHNode::BVHNode(const eastl::vector<Hittable*>& src_objects, size_t start, size_t end, double time0,
	double time1)
		: Hittable(this)
{

	eastl::vector<Hittable*> objects; // Create a modifiable array of the source scene objects
	for (auto src : src_objects)
	{
		objects.push_back(src);
	}

	auto seed = (uint32_t)rand();
	int axis = RandomInt(seed, 0, 2);
	auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

	size_t object_span = end - start;

	static size_t one = 0, two = 0, more = 0;
	if (object_span == 1) {
		m_Left = m_Right = objects[start];
		one++;
	}
	else if (object_span == 2) {
		if (comparator(objects[start], objects[start + 1])) {
			m_Left = objects[start];
			m_Right = objects[start + 1];
		}
		else {
			m_Left = objects[start + 1];
			m_Right = objects[start];
		}
		two++;
	}
	else {
		std::sort(objects.begin() + start, objects.begin() + end, comparator);

		auto mid = start + object_span / 2;
		m_Left =  new BVHNode(objects, start, mid, time0, time1);
		m_Right = new BVHNode(objects, mid, end, time0, time1);
		more++;
	}

	LOG_CORE_INFO("One :{}, Two: {}, More: {}, Axis: {}", one, two, more, axis);

	AABB box_left, box_right;

	if (!m_Left->BoundingBox(time0, time1, box_left) || !m_Right->BoundingBox(time0, time1, box_right))
		LOG_CORE_ERROR("No bounding box in bvh_node constructor.");

	m_Box = SurroundingBox(box_left, box_right);
}


