#define VERSION "0.0.1"

#include "aparse/aparse.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;
typedef int8_t s8;

// parameters
u32 x_bits = 13;
u32 y_bits = 11;
u32 r_bits = 8;
u32 g_bits = 8;
u32 b_bits = 8;
u32 seed = 0xDEADBEEF;

// return number of unique x-coordinates
u32 x_span() { return 1 << x_bits; }
// return number of unique y-coordinates
u32 y_span() { return 1 << y_bits; }
// return number of unique pixel positions
u32 pixels() { return x_span() * y_span(); }

// return number of unique r-coordinates
u32 r_span() { return 1 << r_bits; }
// return number of unique g-coordinates
u32 g_span() { return 1 << g_bits; }
// return number of unique b-coordinates
u32 b_span() { return 1 << b_bits; }
// return number of unique colors
u32 colors() { return r_span() * g_span() * b_span(); }

// compose bitfield
u32 bf_compress(u32 value, u32 base_index, u32 width) {
  return (value & ((1 << width) - 1)) << base_index;
}
// decompose bitfield
u32 bf_extract(u32 value, u32 base_index, u32 width) {
  return (value >> base_index) & ((1 << width) - 1);
}

// sign bit used for packed directions
typedef u8 sign_flag;
#define SIGN_FLAG_POS 0
#define SIGN_FLAG_NEG 1
// convert sign bit [0 = positive, 1 = negative] to 1 or -1, respectively
int sign_flag_to_int(sign_flag sf) { return ((int)sf) * -2 + 1; }

// axes used for moving around the xy rectangle / rgb cube
typedef u8 axis;
#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2
// get span of axis (square)
u32 axis_span_square(axis ax) { return (ax == AXIS_X) ? x_span() : y_span(); }
// get span of axis (cube)
u32 axis_span_cube(axis ax) {
  return (ax == AXIS_X) ? r_span() : (ax == AXIS_Y) ? g_span() : b_span();
}

// bitset of multiple axes
typedef u8 axis_flag;
#define AXIS_FLAG_X axis_to_axis_flag(AXIS_X)
#define AXIS_FLAG_Y axis_to_axis_flag(AXIS_Y)
#define AXIS_FLAG_Z axis_to_axis_flag(AXIS_Z)
// convert axis to an axis flag (bit index)
axis_flag axis_to_axis_flag(axis ax) { return 1 << ax; }

// a direction is just an axis combined with a sign
typedef u8 dir;
#define DIR_PX dir_make(SIGN_FLAG_POS, AXIS_X)
#define DIR_NX dir_make(SIGN_FLAG_NEG, AXIS_X)
#define DIR_PY dir_make(SIGN_FLAG_POS, AXIS_Y)
#define DIR_NY dir_make(SIGN_FLAG_NEG, AXIS_Y)
#define DIR_PZ dir_make(SIGN_FLAG_POS, AXIS_Z)
#define DIR_NZ dir_make(SIGN_FLAG_NEG, AXIS_Z)
#define DIR_INVALID (DIR_NZ + 1)

// create dir from sign flag and axis
dir dir_make(sign_flag sf, axis ax) { return ax * 2 + sf; }
// get axis of dir
dir dir_axis(dir d) { return d >> 1; }
// get sign of dir
sign_flag dir_sign_flag(u8 d) { return d & 1; }
// invert sign of dir
dir dir_invert(dir d) { return dir_make(!dir_sign_flag(d), dir_axis(d)); }
// convert dir to string
const char *dir_to_str(dir d) {
  return (const char *[]){"+x", "-x", "+y", "-y", "+z", "-z"}[d];
}

// bitset of combined directions
typedef u8 dir_flag;
#define DIR_FLAG_PX dir_to_dir_flag(DIR_PX)
#define DIR_FLAG_NX dir_to_dir_flag(DIR_NX)
#define DIR_FLAG_PY dir_to_dir_flag(DIR_PY)
#define DIR_FLAG_NY dir_to_dir_flag(DIR_NY)
#define DIR_FLAG_PZ dir_to_dir_flag(DIR_PZ)
#define DIR_FLAG_NZ dir_to_dir_flag(DIR_NZ)
// convert dir to dir bitset
dir_flag dir_to_dir_flag(dir d) { return 1 << d; }

// point on the xy square
typedef u32 sq_point;
// half-point on the xy square
typedef u32 sq_hpoint;
// point on the rgb cube
typedef u32 cb_point;
// half-point on the rgb cube
typedef u32 cb_hpoint;

// make square point from x and y coordinates
sq_point sq_point_make(u32 x, u32 y) {
  return bf_compress(x, 0, x_bits) | bf_compress(y, x_bits, y_bits);
}
// make square half point from x and y coordinates
sq_hpoint sq_hpoint_make(u32 x, u32 y) {
  return bf_compress(x >> 1, 0, x_bits - 1) |
         bf_compress(y >> 1, x_bits - 1, y_bits - 1);
}

// get x coordinate of square point
u32 sq_point_x(sq_point pt) { return bf_extract(pt, 0, x_bits); }
// get x coordinate of square half point
u32 sq_hpoint_x(sq_hpoint pt) { return bf_extract(pt, 0, x_bits - 1) << 1; }

// get y coordinate of square point
u32 sq_point_y(sq_point pt) { return bf_extract(pt, x_bits, y_bits); }
// get y coordinate of square half point
u32 sq_hpoint_y(sq_hpoint pt) {
  return bf_extract(pt, x_bits - 1, y_bits - 1) << 1;
}
// add direction to square point
u32 sq_point_add_dir(sq_point pt, dir d) {
  int sign = sign_flag_to_int(dir_sign_flag(d));
  return sq_point_make(sq_point_x(pt) + (dir_axis(d) == AXIS_X) * sign,
                       sq_point_y(pt) + (dir_axis(d) == AXIS_Y) * sign);
}
// add direction to square half point
sq_hpoint sq_hpoint_add_dir(sq_hpoint pt, dir d) {
  int sign = sign_flag_to_int(dir_sign_flag(d));
  return sq_hpoint_make(sq_hpoint_x(pt) + (dir_axis(d) == AXIS_X) * sign * 2,
                        sq_hpoint_y(pt) + (dir_axis(d) == AXIS_Y) * sign * 2);
}
// get axis component of square point
u32 sq_point_get_axis(sq_point pt, axis ax) {
  return (ax == AXIS_X) ? sq_point_x(pt) : sq_point_y(pt);
}
// get axis component of square half point
u32 sq_hpoint_get_axis(sq_hpoint pt, axis ax) {
  return (ax == AXIS_X) ? sq_hpoint_x(pt) : sq_hpoint_y(pt);
}

// make cube point from r, g, and b coordinates
cb_point cb_point_make(u32 r, u32 g, u32 b) {
  return bf_compress(r, 0, r_bits) | bf_compress(g, r_bits, g_bits) |
         bf_compress(b, r_bits + g_bits, b_bits);
}
// make cube half point from r, g, and b coordinates
cb_hpoint cb_hpoint_make(u32 r, u32 g, u32 b) {
  return bf_compress(r >> 1, 0, r_bits - 1) |
         bf_compress(g >> 1, r_bits - 1, g_bits - 1) |
         bf_compress(b >> 1, (r_bits - 1 + g_bits - 1), b_bits - 1);
}

// get r coordinate of cube point
u32 cb_point_r(cb_point pt) { return bf_extract(pt, 0, r_bits); }
// get r coordinate of cube half point
u32 cb_hpoint_r(cb_hpoint pt) { return bf_extract(pt, 0, r_bits - 1) << 1; }
// get g coordinate of cube point
u32 cb_point_g(cb_point pt) { return bf_extract(pt, r_bits, g_bits); }
// get g coordinate of cube half point
u32 cb_hpoint_g(cb_hpoint pt) {
  return bf_extract(pt, r_bits - 1, g_bits - 1) << 1;
}
// get b coordinate of cube point
u32 cb_point_b(cb_point pt) { return bf_extract(pt, r_bits + g_bits, b_bits); }
// get b coordinate of cube half point
u32 cb_hpoint_b(cb_hpoint pt) {
  return bf_extract(pt, r_bits - 1 + g_bits - 1, b_bits - 1) << 1;
}
// add direction to cube point
u32 cb_point_add_dir(cb_point pt, dir d) {
  int sign = sign_flag_to_int(dir_sign_flag(d));
  return cb_point_make(cb_point_r(pt) + (dir_axis(d) == AXIS_X) * sign,
                       cb_point_g(pt) + (dir_axis(d) == AXIS_Y) * sign,
                       cb_point_b(pt) + (dir_axis(d) == AXIS_Z) * sign);
}
// add direction to cube half point
u32 cb_hpoint_add_dir(cb_hpoint pt, dir d) {
  int sign = sign_flag_to_int(dir_sign_flag(d));
  return cb_hpoint_make(cb_hpoint_r(pt) + (dir_axis(d) == AXIS_X) * sign * 2,
                        cb_hpoint_g(pt) + (dir_axis(d) == AXIS_Y) * sign * 2,
                        cb_hpoint_b(pt) + (dir_axis(d) == AXIS_Z) * sign * 2);
}
// get axis component of cube point
u32 cb_point_get_axis(cb_point pt, axis ax) {
  return (ax == AXIS_X)   ? cb_point_r(pt)
         : (ax == AXIS_Y) ? cb_point_g(pt)
                          : cb_point_b(pt);
}
// get axis component of cube half point
u32 cb_hpoint_get_axis(cb_hpoint pt, axis ax) {
  return (ax == AXIS_X)   ? cb_hpoint_r(pt)
         : (ax == AXIS_Y) ? cb_hpoint_g(pt)
                          : cb_hpoint_b(pt);
}

// node id for MST generation purposes
typedef u32 node_id;
typedef struct edge {
  node_id from;
  node_id to;
} edge;

// make an edge going from `from` to `to`
edge edge_make(node_id from, node_id to) {
  edge out;
  out.from = from;
  out.to = to;
  return out;
}

// mergesort edges weighted by weights
void edges_sort(edge *edges, u32 *weights, u32 from, u32 to, edge *edges_target,
                u32 *weights_target) {
  if (to - from <= 1)
    return;
  u32 mid = (from + to) / 2;
  edges_sort(edges_target, weights_target, from, mid, edges, weights);
  edges_sort(edges_target, weights_target, mid, to, edges, weights);
  u32 left = from;
  u32 right = mid;
  u32 i = left;
  for (; i < to && left < mid && right < to; i++) {
    if (weights[left] <= weights[right]) {
      weights_target[i] = weights[left];
      edges_target[i] = edges[left];
      left++;
    } else {
      weights_target[i] = weights[right];
      edges_target[i] = edges[right];
      right++;
    }
  }
  if (left < mid) {
    memmove(&weights_target[i], &weights[left], (mid - left) * sizeof(u32));
    memmove(&edges_target[i], &edges[left], (mid - left) * sizeof(edge));
  } else if (right < to) {
    memmove(&weights_target[i], &weights[right], (to - right) * sizeof(u32));
    memmove(&edges_target[i], &edges[right], (to - right) * sizeof(edge));
  }
}

// union-find acceleration datastructure
typedef struct dsu {
  node_id *parent;
  u32 *rank;
} dsu;

// initialize dsu
void dsu_init(dsu *d, node_id max_node) {
  d->parent = malloc(sizeof(node_id) * max_node);
  d->rank = malloc(sizeof(u32) * max_node);
  assert(d->parent && d->rank);
  for (node_id i = 0; i < max_node; i++)
    d->parent[i] = i, d->rank[i] = 1;
}

// destroy dsu
void dsu_destroy(dsu *d) {
  free(d->parent);
  free(d->rank);
}

// find root of given node
node_id dsu_find(dsu *d, node_id node) {
  node_id root = node;
#define DSU_JIT_METHOD 0
#if DSU_JIT_METHOD == 0
  // path compression
  while (d->parent[root] != root)
    // drill down the DSU to find the root node (representative set)
    root = d->parent[root];
  while (d->parent[node] != root) {
    // shorten node->parent chains to accelerate subsequent lookups
    node_id parent = d->parent[node];
    d->parent[node] = root;
    node = parent;
  }
#elif DSU_JIT_METHOD == 1
  // path splitting
  while (d->parent[root] != root) {
    node_id parent = d->parent[root], grandparent = d->parent[parent];
    d->parent[root] = parent, d->parent[parent] = grandparent;
    root = parent;
  }
#elif DSU_JIT_METHOD == 2
  // path halving
  while (d->parent[root] != root) {
    node_id parent = d->parent[root], grandparent = d->parent[parent];
    d->parent[parent] = grandparent;
    d->parent[root] = grandparent;
    root = grandparent;
  }
#endif
  return root;
}

// union the representative sets of two root nodes
void dsu_root_union(dsu *d, node_id root_x, node_id root_y) {
  // for max efficiency we should never call this with root_x == root_y
  assert(root_x != root_y);
  if (d->rank[root_x] < d->rank[root_y]) {
    // keep trees short by grafting larger trees onto smaller ones
    node_id temp = root_x;
    root_x = root_y;
    root_y = temp;
  }
  d->parent[root_y] = root_x;
  if (d->rank[root_x] == d->rank[root_y])
    d->rank[root_x] += 1;
}

// kruskal's implementation
edge *kruskinate(u32 num_nodes, u32 num_edges, edge *edges, u32 *weights) {
  dsu d;
  dsu_init(&d, num_nodes);
  edge *edges_sorted = malloc(sizeof(edge) * num_edges);
  u32 *weights_sorted = malloc(sizeof(u32) * num_edges);
  edge *out_edges = malloc(sizeof(edge) * (num_nodes - 1));
  assert(edges_sorted && weights_sorted && out_edges);
  // need to copy into edges_sorted/weights_sorted for base case
  memcpy(edges_sorted, edges, sizeof(edge) * num_edges);
  memcpy(weights_sorted, weights, sizeof(u32) * num_edges);
  edges_sort(edges, weights, 0, num_edges, edges_sorted, weights_sorted);
  // delete old edges
  free(edges);
  free(weights);
  u32 pop_idx = num_edges;
  u32 tree_num_edges = 0;
  u32 tree_max_edges = num_nodes - 1;
  while (tree_num_edges != tree_max_edges) {
    assert(pop_idx);
    edge next_edge = edges_sorted[--pop_idx];
    node_id root_a = dsu_find(&d, next_edge.from);
    node_id root_b = dsu_find(&d, next_edge.to);
    if (root_a != root_b) {
      dsu_root_union(&d, root_a, root_b);
      out_edges[tree_num_edges++] = next_edge;
    }
  }
  dsu_destroy(&d);
  free(edges_sorted);
  free(weights_sorted);
  return out_edges;
}

// generate all initial square half point edges
edge *sq_edges_make(u32 w, u32 h, u32 *out_num_edges) {
  assert(!(w % 2) && !(h % 2));
  edge *out_edges = malloc(sizeof(edge) * ((w / 2) * (h / 2) * 2));
  assert(out_edges);
  u32 x, y;
  *out_num_edges = 0;
  for (y = 0; y < h; y += 2) {
    for (x = 0; x < w; x += 2) {
      sq_hpoint this_coord = sq_hpoint_make(x, y);
      if (x)
        // create edge from (x - 2, y) -> (x, y)
        out_edges[(*out_num_edges)++] =
            edge_make(sq_hpoint_add_dir(this_coord, DIR_NX), this_coord);
      if (y)
        // create edge from (x, y - 2) -> (x, y)
        out_edges[(*out_num_edges)++] =
            edge_make(sq_hpoint_add_dir(this_coord, DIR_NY), this_coord);
    }
  }
  return out_edges;
}

// generate all initial cube half point edges
edge *make_cube_edges(u32 w, u32 h, u32 d, u32 *out_num_edges) {
  assert(!(w % 2) && !(h % 2) && !(d % 2));
  edge *out_edges = malloc(sizeof(edge) * ((w / 2) * (h / 2) * (d / 2) * 3));
  assert(out_edges);
  u32 r, g, b;
  *out_num_edges = 0;
  for (b = 0; b < d; b += 2) {
    for (g = 0; g < h; g += 2) {
      for (r = 0; r < w; r += 2) {
        u32 this_coord = cb_hpoint_make(r, g, b);
        if (r)
          // create edge from (r - 2, g, b) -> (r, g, b)
          out_edges[(*out_num_edges)++] =
              edge_make(cb_hpoint_add_dir(this_coord, DIR_NX), this_coord);
        if (g)
          // create edge from (r, g - 2, b) -> (r, g, b)
          out_edges[(*out_num_edges)++] =
              edge_make(cb_hpoint_add_dir(this_coord, DIR_NY), this_coord);
        if (b)
          // create edge from (r, g, b - 2) -> (r, g, b)
          out_edges[(*out_num_edges)++] =
              edge_make(cb_hpoint_add_dir(this_coord, DIR_NZ), this_coord);
      }
    }
  }
  return out_edges;
}

// transform edge list into X/2 by Y/2 array of direction flags
dir_flag *map_square_edges(u32 num_edges, edge *edges) {
  dir_flag *dir_map = malloc(sizeof(u8) * pixels() / 4);
  assert(dir_map);
  memset(dir_map, 0, sizeof(u8) * pixels() / 4);
  while (num_edges--) {
    edge e = edges[num_edges];
    dir_flag dir_mask = 0;
    for (dir i = 0; i < 4; i++) {
      if (e.to == sq_hpoint_add_dir(e.from, i)) {
        dir_mask = dir_to_dir_flag(i);
        break;
      }
    }
    dir_map[e.from] |= dir_mask;
  }
  return dir_map;
}

// transform edge list into R/2 by G/2 by B/2 array of direction flags
dir_flag *map_cube_edges(u32 num_edges, edge *edges) {
  dir_flag *dir_map = malloc(sizeof(u8) * colors() / 8);
  assert(dir_map);
  memset(dir_map, 0, sizeof(u8) * colors() / 8);
  while (num_edges--) {
    edge e = edges[num_edges];
    dir_flag dir_mask = 0;
    for (dir i = 0; i < 6; i++) {
      if (e.to == cb_hpoint_add_dir(e.from, i)) {
        dir_mask = dir_to_dir_flag(i);
        break;
      }
    }
    dir_map[e.from] |= dir_mask;
  }
  return dir_map;
}

// given a starting point on the square, flip edges so that we end up with a
// tree rooted at that point rather than (0, 0).
dir_flag *reorient_square_edges(dir_flag *dir_map, sq_hpoint start_idx) {
  sq_hpoint *stk = malloc(sizeof(u32) * pixels() / 4);
  dir_flag *undir_map = malloc(sizeof(dir_flag) * pixels() / 4);
  assert(stk && undir_map);
  memset(undir_map, 0, sizeof(dir_flag) * pixels() / 4);
  u32 stk_ptr = 0;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    sq_hpoint top = stk[--stk_ptr];
    for (dir i = 0; i < 4; i++) {
      u32 npos;
      dir ndir;
      if (sq_hpoint_get_axis(top, dir_axis(i)) ==
          ((dir_sign_flag(i) == SIGN_FLAG_POS)
               ? axis_span_square(dir_axis(i)) - 1
               : 0))
        continue;
      if (dir_map[npos = sq_hpoint_add_dir(top, i)] &
          (ndir = dir_to_dir_flag(dir_invert(i))))
        dir_map[npos] &= ~ndir, undir_map[top] |= dir_to_dir_flag(i),
            stk[stk_ptr++] = npos;
    }
    for (dir i = 0; i < 4; i++) {
      dir d_flag;
      if (dir_map[top] & (d_flag = dir_to_dir_flag(i)))
        dir_map[top] &= ~d_flag, undir_map[top] |= d_flag,
            stk[stk_ptr++] = sq_hpoint_add_dir(top, i);
    }
  }
  free(dir_map);
  free(stk);
  return undir_map;
}

// given a starting point on the cube, flip edges so that we end up with a tree
// rooted at that point rather than (0, 0, 0).
dir_flag *reorient_cube_edges(dir_flag *dir_map, cb_hpoint start_idx) {
  cb_hpoint *stk = malloc(sizeof(u32) * colors() / 8);
  dir_flag *undir_map = malloc(sizeof(dir_flag) * colors() / 8);
  assert(stk && undir_map);
  memset(undir_map, 0, sizeof(dir_flag) * colors() / 8);
  u32 stk_ptr = 0;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    cb_hpoint top = stk[--stk_ptr];
    for (dir i = 0; i < 6; i++) {
      u32 npos;
      dir ndir;
      if (cb_hpoint_get_axis(top, dir_axis(i)) ==
          ((dir_sign_flag(i) == SIGN_FLAG_POS) ? axis_span_cube(dir_axis(i)) - 1
                                               : 0))
        continue;
      if (dir_map[npos = cb_hpoint_add_dir(top, i)] &
          (ndir = dir_to_dir_flag(dir_invert(i))))
        dir_map[npos] &= ~ndir, undir_map[top] |= dir_to_dir_flag(i),
            stk[stk_ptr++] = npos;
    }
    for (dir i = 0; i < 6; i++) {
      dir d_flag;
      if (dir_map[top] & (d_flag = dir_to_dir_flag(i)))
        dir_map[top] &= ~d_flag, undir_map[top] |= d_flag,
            stk[stk_ptr++] = cb_hpoint_add_dir(top, i);
    }
  }
  free(dir_map);
  free(stk);
  return undir_map;
}

typedef u8 *gray3;

// invert `mask` bits in every element of `g`
void gray3_invert(gray3 g, axis_flag mask) {
  for (int i = 0; i < 8; i++)
    g[i] = ((~g[i] & mask) | (g[i] & ~mask)) & 7;
}

// swap single bits `a_mask` and `b_mask` in every element of `g`
void gray3_swap(gray3 g, axis_flag a_mask, axis_flag b_mask) {
  u8 others = ~(a_mask | b_mask);
  for (int i = 0; i < 8; i++) {
    g[i] = (g[i] & others) | (!!(g[i] & a_mask) * b_mask) |
           (!!(g[i] & b_mask) * a_mask);
  }
}

// popcount of point on gray square
u8 popcount_2(axis_flag point) { return !!(point & 1) + !!(point & 2); }

// popcount of point on gray cube
u8 popcount_3(axis_flag point) {
  return !!(point & 1) + !!(point & 2) + !!(point & 4);
}

// find direction with fewest nonzero amount of remaining splice locations
dir dir_argmin(u32 packed_axis_count) {
  u8 min = 4;
  dir d = 0;
  for (dir i = 0; i < 6; i++) {
    u8 num_remaining_splices = packed_axis_count & ((1 << 2) - 1);
    if (num_remaining_splices && num_remaining_splices < min)
      min = num_remaining_splices, d = i;
    packed_axis_count >>= 2;
  }
  return d;
}

// given bitset of out-edges, and index of in-edge, compute a gray code that
// visits all four positions on the unit square, and additionally the direction
// for each of those positions.
void grayinate_2(dir_flag child_set, dir dir_out, dir *out_dirs,
                 axis_flag *out_gray) {
  // base 2-bit gray code 00 01 11 10
  axis_flag gray[4] = {0, 1, 3, 2};
  // base direction for each pair of the above code
  dir dirs[4] = {DIR_PX, DIR_PY, DIR_NX, DIR_NY};
  // inverse of the above table
  u8 dir_to_idx[4] = {1, 3, 2, 0};
  // copy base directions to out directoins
  for (dir i = 0; i < 4; i++) {
    out_dirs[i] = dirs[i];
  }
  // if this isn't the root square, we have an output direction
  if (dir_out < 6) {
    out_dirs[dir_to_idx[dir_out]] = dir_out;
  }
  // set directions of relevant out-edges
  for (dir i = 0; i < 4; i++) {
    if (child_set & dir_to_dir_flag(i))
      out_dirs[dir_to_idx[i]] = i;
    out_gray[i] = gray[i];
  }
}

// given bitset of out-edges, and index of in-edge, compute a gray code that
// visits all eight positions on the unit cube, and additionally the direction
// for each of those positions.
// Also, the gray code must start and end at given points on the cube.
void grayinate_3(axis_flag start_point, axis_flag end_point, dir_flag child_set,
                 dir dir_out, dir *out_dirs, u8 *out_gray) {
  u8 gray[8] = {0, 1, 3, 2, 6, 7, 5, 4}, start_pc, i, j, k,
     orig_child_set = child_set;
  assert(child_set < (1 << 6));
  // 1. Generate a gray code that visits all positions on the unit cube.
  if ((start_pc = popcount_3(start_point)) > popcount_3(end_point))
    // Inverting the gray code will flip the popcounts in favor of start_point.
    gray3_invert(gray, 7);
  i = 1;
  while (popcount_3(gray[0]) != start_pc) {
    // Invert bits until popcount equals that of start_pc.
    if ((gray[0] & i) == (gray[7] & i))
      gray3_invert(gray, i);
    i <<= 1;
  }
  for (i = 1; i < 4; i <<= 1) {
    for (j = i << 1; j < 8; j <<= 1) {
      // Swap bits until the start point of the gray code matches the input
      // start point.
      if (((gray[0] & i) != (start_point & i)) &&
          (gray[0] & j) != (start_point & j))
        gray3_swap(gray, i, j);
      // Then, swap bits until the end point matches the input end point.
      else if (((gray[7] & i) != (end_point & i)) &&
               (gray[7] & j) != (end_point & j))
        gray3_swap(gray, i, j);
    }
  }
  // 2. Figure out how to assign directions.
  u8 dirs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (i = 0; i < 8; i++) {
    // Build initial direction vectors by diffing pairs of gray points.
    u8 a = gray[i], b = gray[(i + 1) & 7];
    u8 diff = a ^ b;
    u8 sign = !(b & diff); // 0 = pos, 1 = neg
    dirs[i] = !!(diff & 1) * DIR_PX + !!(diff & 2) * DIR_PY +
              !!(diff & 4) * DIR_PZ + sign;
  }
  // Build axis count and axis validity sets.
  u64 axis_count = 0, axis;
  u64 dir_axis_set[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (i = 0; i < 8; i++) {
    u8 dir = dirs[i], point = gray[i], inv = dir_invert(dir);
    for (axis = 0; axis < 3; axis++) {
      u8 sgn = !((1 << axis) & point);
      u8 compat_dir = dir_make(sgn, axis);
      if (compat_dir != inv)
        dir_axis_set[i] |= 1 << (2 * compat_dir);
    }
    axis_count += dir_axis_set[i];
  }
  if (dir_out < 6) {
    axis_count -= dir_axis_set[7]; // splice out end
    dir_axis_set[7] = 0;
    dirs[7] = dir_out;
    for (k = 0; k < 8; k++) {
      // invalidate other dirs
      dir_axis_set[k] &= ~(3 << (2 * dir_out));
    }
  }
  for (i = 0; i < 6 && child_set; i++) {
    u8 current_dir = dir_argmin(axis_count);
    u8 current_dir_flag = dir_to_dir_flag(current_dir);
    u64 dir_mask = 3 << (2 * current_dir), dir_mask_inv = ~dir_mask;
    if (current_dir_flag & child_set) {
      for (j = 0; j < 8; j++) {
        if (dir_axis_set[j] & dir_mask) { // found matching point, splice out
          axis_count -= dir_axis_set[j];
          axis_count &= dir_mask_inv;
          dir_axis_set[j] = 0;
          dirs[j] = current_dir;
          for (k = 0; k < 8; k++) {
            // invalidate other dirs
            dir_axis_set[k] &= dir_mask_inv;
          }
          break;
        }
      }
      child_set &= ~current_dir_flag;
    } else {
      // we don't care about this dir
      for (k = 0; k < 8; k++) {
        dir_axis_set[k] &= dir_mask_inv;
      }
      axis_count &= dir_mask_inv;
    }
  }
  for (i = 0; i < 8; i++) {
    out_dirs[i] = dirs[i];
    out_gray[i] = gray[i];
    assert(gray[i] < 8 && dirs[i] < 6);
  }
  u8 check = orig_child_set | ((dir_out < 6) ? dir_to_dir_flag(dir_out) : 0);
  for (j = 0; j < 8; j++) {
    u8 d = out_dirs[j];
    for (axis = 0; axis < 3; axis++) {
      if (d == dir_make(!(gray[j] & (1 << axis)), axis)) {
        assert((gray[(j + 1) % 8] ^ gray[j]) != (1 << axis));
        assert(check & (1 << d));
        check ^= 1 << d;
      }
    }
  }
  if (dir_out < 6) {
    assert(dirs[7] == dir_out);
  }
  assert(!check);
}

u8 *resolve_edges_2(u32 num_nodes, u8 *dir_map, u32 start_idx) {
  u32 *stk = malloc(sizeof(u32) * num_nodes);
  u8 *dir_out_stk = malloc(sizeof(u8) * num_nodes);
  u8 *out = malloc(sizeof(u8) * num_nodes * 4);
  assert(stk && dir_out_stk && out);
  memset(out, 6, sizeof(u8) * num_nodes * 4);
  u32 stk_ptr = 0;
  dir_out_stk[stk_ptr] = 6;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    u32 top = stk[stk_ptr - 1];
    u8 dir = dir_map[top], dirs[4], dir_out = dir_out_stk[stk_ptr - 1], gray[4];
    u32 i;
    stk_ptr--;
    grayinate_2(dir, dir_out, dirs, gray);
    u32 idx = sq_point_make(sq_hpoint_x(top), sq_hpoint_y(top));
    for (i = 0; i < 4; i++) {
      u8 gray_point = gray[i];
      u32 out_idx = sq_point_make(sq_point_x(idx) + (gray_point & 1),
                                  sq_point_y(idx) + !!(gray_point & 2));
      assert(out[out_idx] == 6);
      out[out_idx] = dirs[i];
    }
    for (i = 0; i < 4; i++) {
      if (dir & dir_to_dir_flag(i)) {
        dir_out_stk[stk_ptr] = dir_invert(i);
        stk[stk_ptr++] = sq_hpoint_add_dir(top, i);
      }
    }
  }
  free(dir_out_stk);
  free(stk);
  return out;
}

u8 *resolve_edges_3(u32 num_nodes, u8 *dir_map, u32 start_idx) {
  u32 *stk = malloc(sizeof(u32) * num_nodes);
  u8 *dir_out_stk = malloc(sizeof(u8) * num_nodes);
  u8 *start_point_stk = malloc(sizeof(u8) * num_nodes);
  u8 *end_point_stk = malloc(sizeof(u8) * num_nodes);
  u8 *out = malloc(sizeof(u8) * num_nodes * 8);
  assert(stk && dir_out_stk && start_point_stk && end_point_stk && out);
  memset(out, 7, sizeof(u8) * num_nodes * 8);
  u32 stk_ptr = 0;
  // u32 idxidx = 0;
  dir_out_stk[stk_ptr] = 6;
  start_point_stk[stk_ptr] = 0;
  end_point_stk[stk_ptr] = 1;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    u32 top = stk[stk_ptr - 1];
    u8 dir = dir_map[top], dirs[8], dir_out = dir_out_stk[stk_ptr - 1], gray[8];
    u8 start_point = start_point_stk[stk_ptr - 1],
       end_point = end_point_stk[stk_ptr - 1];
    u32 i;
    stk_ptr--;
    grayinate_3(start_point, end_point, dir, dir_out, dirs, gray);
    u32 idx =
        cb_point_make(cb_hpoint_r(top), cb_hpoint_g(top), cb_hpoint_b(top));
    for (i = 0; i < 8; i++) {
      u8 gray_point = gray[i];
      assert(dirs[i] < 6);
      u32 out_idx = cb_point_make(cb_point_r(idx) + (gray_point & 1),
                                  cb_point_g(idx) + !!(gray_point & 2),
                                  cb_point_b(idx) + !!(gray_point & 4));
      assert(out[out_idx] == 7);
      out[out_idx] = dirs[i];
      u8 dir_mask = (1 << (dirs[i] >> 1));
      if (((top == start_idx) || i < 7) &&
          !!(gray_point & dir_mask) ^ (dirs[i] & 1)) {
        start_point_stk[stk_ptr] = gray_point ^ dir_mask;
        end_point_stk[stk_ptr] = gray[(i + 1) % 8] ^ dir_mask;
        dir_out_stk[stk_ptr] = dir_invert(dirs[i]);
        stk[stk_ptr++] = cb_hpoint_add_dir(top, dirs[i]);
      }
    }
    // if ((++idxidx % 100000) == 0 || idxidx == num_nodes)
    //   printf("resolve: %u\n", idxidx);
  }
  free(dir_out_stk);
  free(stk);
  free(start_point_stk);
  free(end_point_stk);
  return out;
}

// exact bias: 0.020888578919738908
uint32_t triple32(uint32_t x) {
  x ^= x >> 17;
  x *= 0xed5ad4bbU;
  x ^= x >> 11;
  x *= 0xac4c1b51U;
  x ^= x >> 15;
  x *= 0x31848babU;
  x ^= x >> 14;
  return x;
}

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t *rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

pcg32_random_t rng_state = {0xDEADBEEF, 1};

uint32_t rng(void) { return pcg32_random_r(&rng_state); }

typedef u32 (*weight_func)(edge *e);

u32 *make_edge_weights(u32 num_edges, edge *edges, weight_func func) {
  u32 *out = malloc(sizeof(u32) * num_edges);
  assert(out);
  for (u32 i = 0; i < num_edges; i++) {
    out[i] = func(edges + i);
  }
  return out;
}

u32 rng_weight_func(edge *e) {
  if (rng() & 1) {
    u32 temp = e->from;
    e->from = e->to;
    e->to = temp;
  }
  return rng();
}

u32 quant_weight_func(edge *e) {
  if (rng() & 1) {
    u32 temp = e->from;
    e->from = e->to;
    e->to = temp;
  }
  u32 rv = rng();
  if (rv & 1) {
    return 1 << 31;
  } else {
    return rv;
  }
  return (rng() % 64) == 63;
}

u32 circle_weight_func(edge *e) {
  if (rng() & 1) {
    u32 temp = e->from;
    e->from = e->to;
    e->to = temp;
  }
  u32 idx = e->from;
  float x = sq_hpoint_x(idx), y = sq_hpoint_y(idx);
  x = ((float)x_span() / 2) - x, y = ((float)y_span() / 2) - y;
  float dist = (x * x + y * y);
  if (dist < 1000.0f * 1000.0f) {
    return (rng() % 32) > 17;
  } else {
    return rng() + 10000;
  }
}

u32 whirlpool_weight_func(edge *e) {
  if (rng() & 1) {
    u32 temp = e->from;
    e->from = e->to;
    e->to = temp;
  }
  static const float shells = 8;
  static const float swirls = 16;
  static const float swirl_vel = 8;
  float w = (float)x_span() / 2, h = (float)y_span() / 2;
  float x = w - sq_hpoint_x(e->from), y = h - sq_hpoint_y(e->from);
  float maxdim = fmaxf(w, h);
  float dist = sqrtf(x * x + y * y) / sqrtf(maxdim * maxdim * 2);
  float ang = fmodf(atan2f(y, x) + 2 * M_PI, 2 * M_PI);
  float phase = dist + (1 / shells / 2) *
                           cosf(ang * swirls + dist * M_PI * 2 * swirl_vel);
  float bias =
      powf((-cosf(phase * 2 * M_PI * shells) + 1) / 2, 4) * 0.999 + 0.01;
  // yes, i know this is not how you are supposed to get random floats.
  return (u32)((float)(rng()) * bias);
}

#include <unistd.h>
int quiet = 0;

void out(const char *text) {
  if (!quiet)
    fprintf(stderr, "%s", text);
}

int check = 0;

u8 *bmp_new(u32 num_elements) {
  u8 *out = calloc((num_elements + 7) / 8, 1);
  return out;
}

void bmp_set(u8 *bmp, u32 idx, int v) {
  u8 mask = 1 << (idx & 7);
  bmp[idx / 8] = (bmp[idx / 8] & ~mask) | (v * mask);
}

int bmp_get(u8 *bmp, u32 idx) { return !!(bmp[idx / 8] & (1 << (idx & 7))); }

void run_pic(u8 *screen_dirs, u8 *cube_dirs, u32 screen_idx, u32 cube_idx,
             FILE *f) {
  u32 orig_cube_idx = cube_idx;
  u32 *pix = malloc(sizeof(u32) * x_span() * y_span());
  u8 *out_f = malloc(sizeof(u8) * x_span() * y_span() * 3);
  u32 max_bpp = r_bits > g_bits ? r_bits : g_bits;
  u8 *check_bmp = check ? bmp_new(x_span() * y_span()) : NULL;
  max_bpp = b_bits > max_bpp ? b_bits : max_bpp;
  max_bpp = 8;
  assert(pix && out_f);
  u32 out_f_ptr = 0;
  fprintf(f, "P6\n%i %i\n%u\n", x_span(), y_span(), (1 << max_bpp) - 1);
  do {
    screen_idx = sq_point_add_dir(screen_idx, screen_dirs[screen_idx]);
    cube_idx = cb_point_add_dir(cube_idx, cube_dirs[cube_idx]);
    pix[screen_idx] = cube_idx;
  } while (cube_idx != orig_cube_idx);
  for (u32 i = 0; i < x_span() * y_span(); i++) {
    u32 cube_idx = pix[i];
    u8 r = cb_point_r(cube_idx), g = cb_point_g(cube_idx),
       b = cb_point_b(cube_idx);
    r = (r << (max_bpp - r_bits));
    g = (g << (max_bpp - g_bits));
    b = (b << (max_bpp - b_bits));
    out_f[out_f_ptr++] = r;
    out_f[out_f_ptr++] = g;
    out_f[out_f_ptr++] = b;
    if (check)
      bmp_set(check_bmp, cube_idx, 1);
  }
  if (check) {
    for (u32 i = 0; i < x_span() * y_span(); i++)
      /* if this fails, we don't make all colors. */
      if (!bmp_get(check_bmp, i))
        out("rgb check failed");
  }
  fwrite(out_f, out_f_ptr, 1, f);
  fclose(f);
  free(out_f);
  free(pix);
  free(check_bmp);
}

int is_pos_pow2(int v) {
  // https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
  return v > 0 && (v & (v - 1)) == 0;
}

u32 int_log2(int v) {
  // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious
  u32 out = 0;
  while (v >>= 1)
    out++;
  return out;
}

int main(int argc, const char *const *argv) {
  int w = 4096, h = 4096, r = 256, g = 256, b = 256, in_seed = 0;
  const char *out_pbm = NULL;
  ap *parser = ap_init(argv[0]);
  ap_opt(parser, 'x', "width");
  ap_type_int(parser, &w);
  ap_help(parser, "width of output image, must be a power of 2 (default 4096)");

  ap_opt(parser, 'y', "height");
  ap_type_int(parser, &h);
  ap_help(parser,
          "height of output image, must be a power of 2 (default 4096)");

  ap_opt(parser, 'r', "red");
  ap_type_int(parser, &r);
  ap_help(parser,
          "number of discrete red values, must be a power of 2 (default 256)");

  ap_opt(parser, 'g', "green");
  ap_type_int(parser, &g);
  ap_help(
      parser,
      "number of discrete green values, must be a power of 2 (default 256)");

  ap_opt(parser, 'b', "blue");
  ap_type_int(parser, &b);
  ap_help(parser,
          "number of discrete blue values, must be a power of 2 (default 256)");

  ap_pos(parser, "FILE");
  ap_type_str(parser, &out_pbm);
  ap_help(parser, "output .pbm file (use -- for stdout)");

  ap_opt(parser, 'q', "quiet");
  ap_type_flag(parser, &quiet);
  ap_help(parser, "don't output diagnostic messages");

  ap_opt(parser, 's', "seed");
  ap_type_int(parser, &in_seed);
  ap_help(parser, "choose a seed for the random number generator (default 0)");

  ap_opt(parser, 'c', "check");
  ap_type_flag(parser, &check);
  ap_help(parser, "ensure that all colors are present");

  ap_opt(parser, 'h', "help");
  ap_type_help(parser);

  ap_opt(parser, 'v', "version");
  ap_type_version(parser, "hamrgb " VERSION);

  ap_description(parser, "build images with all possible colors and positions "
                         "in respective hamiltonian cycles");
  ap_epilog(parser,
            "This program, given parameters about the dimensions of an image "
            "and RGB cube, will find a hamiltonian cycle through the positions "
            "of the output image and a hamiltonian cycle through the positions "
            "of the RGB cube, and subsequently walk through the two at the "
            "same time in order to create an aesthetically pleasing image.\n"
            "Copyright (c)2023 Max Nurzia");

  switch (ap_parse(parser, argc - 1, argv + 1)) {
  case 0:
    break;
  case AP_ERR_EXIT:
    exit(0);
  default:
    exit(1);
  }

  if (!is_pos_pow2(w) || !is_pos_pow2(h) || !is_pos_pow2(r) ||
      !is_pos_pow2(g) || !is_pos_pow2(b))
    exit(ap_error(
        parser,
        "--width, --height, --red, --green, and --blue must all reference "
        "positive nonzero integer values that are multiples of 2"));
  x_bits = int_log2(w);
  y_bits = int_log2(h);
  r_bits = int_log2(r);
  g_bits = int_log2(g);
  b_bits = int_log2(b);
  if (x_bits + y_bits != r_bits + g_bits + b_bits)
    exit(ap_error(parser, "number of pixels (--width * --height) must be equal "
                          "to number of colors (--red * --green * --blue)"));

  FILE *out_file = strcmp(out_pbm, "--") ? fopen(out_pbm, "w") : stdout;

  rng_state.state = (u32)in_seed;

  u32 num_screen_edges, num_cube_edges;
  u32 num_screen_nodes = x_span() / 2 * y_span() / 2;
  u32 num_cube_nodes = r_span() / 2 * g_span() / 2 * b_span() / 2;
  sq_hpoint screen_start =
      sq_hpoint_make(rng() & (x_span() - 1), rng() & (y_span() - 1));
  cb_hpoint cube_start = cb_hpoint_make(
      rng() & (r_span() - 1), rng() & (g_span() - 1), rng() & (b_span() - 1));

  out("making edges...\n");
  edge *screen_edges = sq_edges_make(x_span(), y_span(), &num_screen_edges);
  edge *cube_edges =
      make_cube_edges(r_span(), g_span(), b_span(), &num_cube_edges);
  out("making edge weights...\n");
  u32 *screen_edge_weights =
      make_edge_weights(num_screen_edges, screen_edges, &rng_weight_func);
  u32 *cube_edge_weights =
      make_edge_weights(num_cube_edges, cube_edges, &rng_weight_func);
  out("kruskinating...\n");
  screen_edges = kruskinate(num_screen_nodes, num_screen_edges, screen_edges,
                            screen_edge_weights);
  cube_edges =
      kruskinate(num_cube_nodes, num_cube_edges, cube_edges, cube_edge_weights);
  out("mapping edges...\n");
  u8 *screen_edge_dirs = map_square_edges(num_screen_nodes - 1, screen_edges);
  u8 *cube_edge_dirs = map_cube_edges(num_cube_nodes - 1, cube_edges);
  out("reorienting edges...\n");
  screen_edge_dirs = reorient_square_edges(screen_edge_dirs, screen_start);
  cube_edge_dirs = reorient_cube_edges(cube_edge_dirs, cube_start);
  out("resolving edges...\n");
  u8 *out_screen_dirs =
      resolve_edges_2(num_screen_nodes, screen_edge_dirs, screen_start);
  u8 *out_cube_dirs =
      resolve_edges_3(num_cube_nodes, cube_edge_dirs, cube_start);
  run_pic(out_screen_dirs, out_cube_dirs, screen_start, cube_start, out_file);
  return 0;
}
