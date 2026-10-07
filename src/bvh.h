#pragma once

#include "rtweekend_utils.h"

#include "aabb.h"
#include "assets.h"

#include <algorithm>
#include <vector>

__host__ static bool box_compare(const Primitive& a, const Primitive& b, int axis_index)
{
	interval a_axis_interval = a.bounding_box().axis_interval(axis_index);
	interval b_axis_interval = b.bounding_box().axis_interval(axis_index);
	return a_axis_interval.min < b_axis_interval.min;
}

__host__ static bool box_x_compare(const Primitive& a, const Primitive& b)
{
	return box_compare(a, b, 0);
}

__host__ static bool box_y_compare(const Primitive& a, const Primitive& b)
{
	return box_compare(a, b, 1);
}

__host__ static bool box_z_compare(const Primitive& a, const Primitive& b)
{
	return box_compare(a, b, 2);
}

// Build the BVH tree on CPU
inline int build_bvh_recursive(std::vector<Primitive>& objects, size_t start, size_t end, std::vector<BVHNode>& out_nodes)
{
	if (end <= start) return -1;

	aabb bbox = aabb::empty();
	for (size_t obj_idx = start; obj_idx < end; obj_idx++)
		bbox = aabb(bbox, objects[obj_idx].bounding_box());

	int axis = bbox.longest_axis();

	auto comparator = (axis == 0) ? box_x_compare
		: (axis == 1) ? box_y_compare
		: box_z_compare;

	size_t object_span = end - start;
	int current_node_idx = static_cast<int>(out_nodes.size());

	out_nodes.push_back(BVHNode{});

	if (object_span == 1)
	{
		out_nodes[current_node_idx].bbox = bbox;
		out_nodes[current_node_idx].left = -1;
		out_nodes[current_node_idx].right = -1;
		out_nodes[current_node_idx].object_index = objects[start].index;
		out_nodes[current_node_idx].type = objects[start].type;
	}
	else if (object_span == 2)
	{
		std::sort(objects.begin() + start, objects.begin() + end, comparator);

		int left_idx = build_bvh_recursive(objects, start, start + 1, out_nodes);
		int right_idx = build_bvh_recursive(objects, start + 1, end, out_nodes);

		out_nodes[current_node_idx].bbox = bbox;
		out_nodes[current_node_idx].left = left_idx;
		out_nodes[current_node_idx].right = right_idx;
		out_nodes[current_node_idx].axis = axis;
		out_nodes[current_node_idx].object_index = -1;
	}
	else
	{
		std::sort(objects.data() + start, objects.data() + end, comparator);

		size_t mid = start + object_span / 2;
		int left_idx = build_bvh_recursive(objects, start, mid, out_nodes);
		int right_idx = build_bvh_recursive(objects, mid, end, out_nodes);

		out_nodes[current_node_idx].bbox = bbox;
		out_nodes[current_node_idx].left = left_idx;
		out_nodes[current_node_idx].right = right_idx;
		out_nodes[current_node_idx].axis = axis;
		out_nodes[current_node_idx].object_index = -1;
	}

	return current_node_idx;
}
