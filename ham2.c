#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;
typedef int8_t s8;

u32 r_bits = 8;
u32 g_bits = 8;
u32 b_bits = 8;

// return number of unique r-coordinates
u32 r_span() { return 1 << r_bits; }
// return number of unique g-coordinates
u32 g_span() { return 1 << g_bits; }
// return number of unique b-coordinates
u32 b_span() { return 1 << b_bits; }

u32 x_bits = 12;
u32 y_bits = 12;

// return number of unique x-coordinates
u32 x_span() { return 1 << x_bits; }
// return number of unique y-coordinates
u32 y_span() { return 1 << y_bits; }

// compose bitfield
u32 bitfield_compress(u32 value, u32 base_index, u32 width) {
  return (value & ((1 << width) - 1)) << base_index;
}

// decompose bitfield
u32 bitfield_extract(u32 value, u32 base_index, u32 width) {
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

// half-point on the xy square
typedef u32 sq_hpoint;

// make square half-point from x and y coordinates
sq_hpoint sq_hpoint_make(u32 x, u32 y) {
  return bitfield_compress(x >> 1, 0, x_bits - 1) |
         bitfield_compress(y >> 1, x_bits - 1, y_bits - 1);
}
// get x coordinate of half-point
u32 sq_hpoint_x(sq_hpoint pt) {
  return bitfield_extract(pt, 0, x_bits - 1) << 1;
}
// get y coordinate of half-point
u32 sq_hpoint_y(sq_hpoint pt) {
  return bitfield_extract(pt, x_bits - 1, y_bits - 1) << 1;
}

// point on the xy square
typedef u32 sq_point;
// make square point from x and y coordinates
sq_point sq_point_make(sq_point x, sq_point y) {
  return bitfield_compress(x, 0, x_bits) | bitfield_compress(y, x_bits, y_bits);
}
// get x coordinate of point
u32 sq_point_x(sq_point idx) { return bitfield_extract(idx, 0, x_bits); }
// get y coordinate of point
u32 sq_point_y(sq_point idx) { return bitfield_extract(idx, x_bits, y_bits); }

// half-point on the rgb cube
typedef u32 cb_hpoint;
// make cube half-point from r, g, and b coordinates
cb_hpoint cb_hpoint_make(u32 r, u32 g, u32 b) {
  return bitfield_compress(r >> 1, 0, r_bits - 1) |
         bitfield_compress(g >> 1, r_bits - 1, g_bits - 1) |
         bitfield_compress(b >> 1, (r_bits - 1 + g_bits - 1), b_bits - 1);
}
// get r coordinate of half point
u32 cb_hpoint_r(cb_hpoint sidx) {
  return bitfield_extract(sidx, 0, r_bits - 1) << 1;
}
// get g coordinate of half point
u32 cb_hpoint_g(cb_hpoint sidx) {
  return bitfield_extract(sidx, r_bits - 1, g_bits - 1) << 1;
}
// get b coordinate of half point
u32 cb_hpoint_b(cb_hpoint sidx) {
  return bitfield_extract(sidx, r_bits - 1 + g_bits - 1, b_bits - 1) << 1;
}

// point on the rgb cube
typedef u32 cb_point;

// make cube point from r, g, and b coordinates
cb_point cb_point_make(u32 r, u32 g, u32 b) {
  return bitfield_compress(r, 0, r_bits) |
         bitfield_compress(g, r_bits, g_bits) |
         bitfield_compress(b, r_bits + g_bits, b_bits);
}
// get r coordinate of cube point
u32 cb_point_r(cb_point pt) { return bitfield_extract(pt, 0, r_bits); }
// get g coordinate of cube point
u32 cb_point_g(cb_point pt) { return bitfield_extract(pt, r_bits, g_bits); }
// get b coordinate of cube point
u32 cb_point_b(cb_point pt) {
  return bitfield_extract(pt, r_bits + g_bits, b_bits);
}

u32 half_idx_add_dir_2(u32 sidx, u8 dir) {
  int sign = sign_flag_to_int(dir_sign_flag(dir));
  return sq_hpoint_make(
      sq_hpoint_x(sidx) + (dir_axis(dir) == AXIS_X) * sign * 2,
      sq_hpoint_y(sidx) + (dir_axis(dir) == AXIS_Y) * sign * 2);
}
u32 half_idx_add_dir_3(u32 sidx, u8 dir) {
  int sign = sign_flag_to_int(dir_sign_flag(dir));
  return cb_hpoint_make(
      cb_hpoint_r(sidx) + (dir_axis(dir) == AXIS_X) * sign * 2,
      cb_hpoint_g(sidx) + (dir_axis(dir) == AXIS_Y) * sign * 2,
      cb_hpoint_b(sidx) + (dir_axis(dir) == AXIS_Z) * sign * 2);
}

u32 idx_add_dir_2(u32 idx, u8 dir) {
  assert(dir < 6);
  int sign = sign_flag_to_int(dir_sign_flag(dir));
  return sq_point_make(sq_point_x(idx) + (dir_axis(dir) == AXIS_X) * sign,
                       sq_point_y(idx) + (dir_axis(dir) == AXIS_Y) * sign);
}
u32 idx_add_dir_3(u32 idx, u8 dir) {
  int sign = sign_flag_to_int(dir_sign_flag(dir));
  return cb_point_make(cb_point_r(idx) + (dir_axis(dir) == AXIS_X) * sign,
                       cb_point_g(idx) + (dir_axis(dir) == AXIS_Y) * sign,
                       cb_point_b(idx) + (dir_axis(dir) == AXIS_Z) * sign);
}

typedef struct edge {
  u32 from;
  u32 to;
} edge;

int valid_idx(u32 idx) { return idx < 16777216; }

int valid_edge(edge e) { return valid_idx(e.from) && valid_idx(e.to); }

void sort_edges(edge *edges, u32 *weights, u32 from, u32 to, edge *edges_target,
                u32 *weights_target) {
  if (to - from <= 1)
    return;
  u32 mid = (from + to) / 2;
  sort_edges(edges_target, weights_target, from, mid, edges, weights);
  sort_edges(edges_target, weights_target, mid, to, edges, weights);
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

void dsu_init(u32 *dsu, u32 *dsu_size, u32 num_nodes) {
  u32 i;
  for (i = 0; i < num_nodes; i++)
    dsu[i] = i, dsu_size[i] = 1;
}

u32 dsu_find(u32 *dsu, u32 node) {
  u32 root = node;
  while (dsu[root] != root)
    root = dsu[root];
  while (dsu[node] != root) {
    u32 parent = dsu[node];
    dsu[node] = root;
    node = parent;
  }
  return root;
}

void dsu_root_union(u32 *dsu, u32 *dsu_size, u32 root_x, u32 root_y) {
  if (root_x == root_y)
    return;
  if (dsu_size[root_x] < dsu_size[root_y]) {
    u32 temp = root_x;
    root_x = root_y;
    root_y = temp;
  }
  dsu[root_y] = root_x;
  dsu_size[root_x] += dsu_size[root_y];
}

edge *kruskinate(u32 num_nodes, u32 num_edges, edge *edges, u32 *edge_weights) {
  u32 *dsu = malloc(sizeof(u32) * num_nodes);
  u32 *dsu_size = malloc(sizeof(u32) * num_nodes);
  edge *edges_sort = malloc(sizeof(edge) * num_edges);
  u32 *weights_sort = malloc(sizeof(u32) * num_edges);
  edge *out_edges = malloc(sizeof(edge) * (num_nodes - 1));
  assert(dsu && dsu_size && edges_sort && weights_sort && out_edges);
  dsu_init(dsu, dsu_size, num_nodes);
  memcpy(edges_sort, edges, sizeof(edge) * num_edges);
  memcpy(weights_sort, edge_weights, sizeof(u32) * num_edges);
  sort_edges(edges, edge_weights, 0, num_edges, edges_sort, weights_sort);
  edges = edges_sort;
  u32 pop_idx = num_edges;
  u32 tree_num_edges = 0;
  u32 tree_max_edges = num_nodes - 1;
  while (tree_num_edges != tree_max_edges) {
    assert(pop_idx);
    edge next_edge = edges[--pop_idx];
    assert(valid_edge(next_edge));
    u32 root_a = dsu_find(dsu, next_edge.from);
    u32 root_b = dsu_find(dsu, next_edge.to);
    if (root_a != root_b) {
      dsu_root_union(dsu, dsu_size, root_a, root_b);
      out_edges[tree_num_edges++] = next_edge;
    }
  }
  free(dsu);
  free(dsu_size);
  free(edges_sort);
  free(weights_sort);
  return out_edges;
}

edge make_edge(u32 from, u32 to) {
  edge out;
  out.from = from;
  out.to = to;
  return out;
}

edge *make_screen_edges(u32 w, u32 h, u32 *out_num_edges) {
  assert(!(w % 2) && !(h % 2));
  edge *out_edges = malloc(sizeof(edge) * ((w / 2) * (h / 2) * 2));
  assert(out_edges);
  u32 x, y;
  *out_num_edges = 0;
  for (y = 0; y < h; y += 2) {
    for (x = 0; x < w; x += 2) {
      u32 this_coord = sq_hpoint_make(x, y);
      if (x)
        out_edges[(*out_num_edges)++] =
            make_edge(sq_hpoint_make(x - 2, y), this_coord);
      if (y)
        out_edges[(*out_num_edges)++] =
            make_edge(sq_hpoint_make(x, y - 2), this_coord);
    }
  }
  return out_edges;
}

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
          out_edges[(*out_num_edges)++] =
              make_edge(cb_hpoint_make(r - 2, g, b), this_coord);
        if (g)
          out_edges[(*out_num_edges)++] =
              make_edge(cb_hpoint_make(r, g - 2, b), this_coord);
        if (b)
          out_edges[(*out_num_edges)++] =
              make_edge(cb_hpoint_make(r, g, b - 2), this_coord);
      }
    }
  }
  return out_edges;
}

u8 *map_edges(u32 num_nodes, u32 num_edges, edge *edges, int dims) {
  u8 *dir_map = malloc(sizeof(u8) * num_nodes);
  assert(dir_map);
  memset(dir_map, 0, sizeof(u8) * num_nodes);
  while (num_edges--) {
    edge e = edges[num_edges];
    u32 dir = 0;
    if (dims == 2) {
      u32 x0 = sq_hpoint_x(e.from), x1 = sq_hpoint_x(e.to),
          y0 = sq_hpoint_y(e.from), y1 = sq_hpoint_y(e.to);
      if (x1 == x0 + 2)
        dir = DIR_FLAG_PX;
      if (y1 == y0 + 2)
        dir = DIR_FLAG_PY;
    } else if (dims == 3) {
      u32 r0 = cb_hpoint_r(e.from), r1 = cb_hpoint_r(e.to),
          g0 = cb_hpoint_g(e.from), g1 = cb_hpoint_g(e.to),
          b0 = cb_hpoint_b(e.from), b1 = cb_hpoint_b(e.to);
      if (r1 == r0 + 2)
        dir = DIR_FLAG_PX;
      if (g1 == g0 + 2)
        dir = DIR_FLAG_PY;
      if (b1 == b0 + 2)
        dir = DIR_FLAG_PZ;
    }
    dir_map[e.from] |= dir;
  }
  return dir_map;
}

u8 *reorient_edges(u32 num_nodes, u8 *dir_map, u32 start_idx, int dims) {
  u32 *stk = malloc(sizeof(u32) * num_nodes);
  u8 *undir_map = malloc(sizeof(u8) * num_nodes);
  assert(stk && undir_map);
  memset(undir_map, 0, sizeof(u8) * num_nodes);
  u32 stk_ptr = 0;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    u32 top = stk[--stk_ptr];
    if (dims == 2) {
      u32 x = sq_hpoint_x(top), y = sq_hpoint_y(top), nx, ny;
      if (x && (dir_map[nx = sq_hpoint_make(x - 2, y)] & DIR_FLAG_PX))
        dir_map[nx] &= ~DIR_FLAG_PX, undir_map[top] |= DIR_FLAG_NX,
            stk[stk_ptr++] = nx;
      if (y && (dir_map[ny = sq_hpoint_make(x, y - 2)] & DIR_FLAG_PY))
        dir_map[ny] &= ~DIR_FLAG_PY, undir_map[top] |= DIR_FLAG_NY,
            stk[stk_ptr++] = ny;
      if (dir_map[top] & DIR_FLAG_PX)
        dir_map[top] &= ~DIR_FLAG_PX, undir_map[top] |= DIR_FLAG_PX,
            stk[stk_ptr++] = sq_hpoint_make(x + 2, y);
      if (dir_map[top] & DIR_FLAG_PY)
        dir_map[top] &= ~DIR_FLAG_PY, undir_map[top] |= DIR_FLAG_PY,
            stk[stk_ptr++] = sq_hpoint_make(x, y + 2);
    } else if (dims == 3) {
      u32 r = cb_hpoint_r(top), g = cb_hpoint_g(top), b = cb_hpoint_b(top), nr,
          ng, nb;
      if (r && (dir_map[nr = cb_hpoint_make(r - 2, g, b)] & DIR_FLAG_PX))
        dir_map[nr] &= ~DIR_FLAG_PX, undir_map[top] |= DIR_FLAG_NX,
            stk[stk_ptr++] = nr;
      if (g && (dir_map[ng = cb_hpoint_make(r, g - 2, b)] & DIR_FLAG_PY))
        dir_map[ng] &= ~DIR_FLAG_PY, undir_map[top] |= DIR_FLAG_NY,
            stk[stk_ptr++] = ng;
      if (b && (dir_map[nb = cb_hpoint_make(r, g, b - 2)] & DIR_FLAG_PZ))
        dir_map[nb] &= ~DIR_FLAG_PZ, undir_map[top] |= DIR_FLAG_NZ,
            stk[stk_ptr++] = nb;
      if (dir_map[top] & DIR_FLAG_PX)
        dir_map[top] &= ~DIR_FLAG_PX, undir_map[top] |= DIR_FLAG_PX,
            stk[stk_ptr++] = cb_hpoint_make(r + 2, g, b);
      if (dir_map[top] & DIR_FLAG_PY)
        dir_map[top] &= ~DIR_FLAG_PY, undir_map[top] |= DIR_FLAG_PY,
            stk[stk_ptr++] = cb_hpoint_make(r, g + 2, b);
      if (dir_map[top] & DIR_FLAG_PZ)
        dir_map[top] &= ~DIR_FLAG_PZ, undir_map[top] |= DIR_FLAG_PZ,
            stk[stk_ptr++] = cb_hpoint_make(r, g, b + 2);
    }
  }
  free(dir_map);
  free(stk);
  return undir_map;
}

int valid_dir_set(u8 dir_set) { return dir_set < (1 << 6); }

void check_mst(u8 *dir_map, u32 num_nodes, u32 start_idx, int dims) {
  u32 *stk = malloc(sizeof(u32) * num_nodes);
  u8 *found = malloc(sizeof(u8) * num_nodes);
  assert(stk && found);
  memset(found, 0, sizeof(u8) * num_nodes);
  u32 stk_ptr = 0;
  stk[stk_ptr++] = start_idx;
  while (stk_ptr) {
    u32 top = stk[--stk_ptr];
    u8 dir = dir_map[top];
    assert(valid_dir_set(dir));
    assert(!found[top]);
    found[top] = 1;
    if (dims == 2) {
      for (u32 i = 0; i < 4; i++) {
        if (dir & dir_to_dir_flag(i))
          stk[stk_ptr++] = half_idx_add_dir_2(top, i);
      }
    } else if (dims == 3) {
      for (u32 i = 0; i < 6; i++) {
        if (dir & dir_to_dir_flag(i))
          stk[stk_ptr++] = half_idx_add_dir_3(top, i);
      }
    }
  }
  for (u32 i = 0; i < num_nodes; i++) {
    assert(found[i]);
  }
  free(stk);
  free(found);
}

void gray_invert(u8 *g, u8 mask) {
  for (int i = 0; i < 8; i++)
    g[i] = ((~g[i] & mask) | (g[i] & ~mask)) & 7;
}

void gray_swap(u8 *g, u8 a_mask, u8 b_mask) {
  u8 others = ~(a_mask | b_mask);
  for (int i = 0; i < 8; i++) {
    g[i] = (g[i] & others) | (!!(g[i] & a_mask) * b_mask) |
           (!!(g[i] & b_mask) * a_mask);
  }
}

u8 popcount_2(u8 point) { return !!(point & 1) + !!(point & 2); }

u8 popcount(u8 point) { return !!(point & 1) + !!(point & 2) + !!(point & 4); }

const char *point2str(u8 point) {
  return (const char *[]){"000", "001", "010", "011",
                          "100", "101", "110", "111"}[point];
}

u8 dir_argmin(u64 packed_acount) {
  u8 v = 4;
  u8 j = 0;
  for (u8 i = 0; i < 6; i++) {
    u8 pack = packed_acount & ((1 << 2) - 1);
    if (pack && pack < v)
      v = pack, j = i;
    packed_acount >>= 2;
  }
  return j;
}

int is_gray_2(u8 *gray) {
  u8 found = 0;
  for (int i = 0; i < 4; i++) {
    if (!(popcount_2(gray[i] ^ gray[(i + 1) % 4]) == 1))
      return 0;
    if (found & (1 << gray[i]))
      return 0;
    found |= 1 << gray[i];
  }
  return 1;
}

void grayinate_2(u8 child_set, u8 dir_out, u8 *out_dirs, u8 *out_gray) {
  u8 gray[4] = {0, 1, 3, 2}, dirs[4] = {DIR_PX, DIR_PY, DIR_NX, DIR_NY};
  u8 i;
  u8 dir_to_idx[4] = {1, 3, 2, 0};
  for (i = 0; i < 4; i++) {
    out_dirs[i] = dirs[i];
  }
  if (dir_out < 6) {
    out_dirs[dir_to_idx[dir_out]] = dir_out;
  }
  for (i = 0; i < 4; i++) {
    if (child_set & dir_to_dir_flag(i))
      out_dirs[dir_to_idx[i]] = i;
    out_gray[i] = gray[i];
  }
  assert(is_gray_2(out_gray));
}

void print_gray(u8 *g) {
  for (u32 i = 0; i < 8; i++) {
    printf("%s%s", i ? " > " : "", point2str(g[i]));
  }
  printf("\n");
}

void print_bin(u64 b, u8 w) {
  for (u8 i = 0; i < w; i++) {
    if (b & (1 << ((w - i) - 1)))
      printf("1");
    else
      printf("0");
  }
}

int is_gray(u8 *gray) {
  u8 found = 0;
  for (int i = 0; i < 8; i++) {
    if (!(popcount(gray[i] ^ gray[(i + 1) % 8]) == 1))
      return 0;
    if (found & (1 << gray[i]))
      return 0;
    found |= 1 << gray[i];
  }
  return 1;
}

void grayinate_3(u8 start_point, u8 end_point, u8 child_set, u8 dir_out,
                 u8 *out_dirs, u8 *out_gray) {
  u8 gray[8] = {0, 1, 3, 2, 6, 7, 5, 4}, start_pc, i, j, k,
     orig_child_set = child_set;
  assert(child_set < (1 << 6));
  if ((start_pc = popcount(start_point)) > popcount(end_point))
    gray_invert(gray, 7);
  i = 1;
  while (popcount(gray[0]) != start_pc) {
    if ((gray[0] & i) == (gray[7] & i))
      gray_invert(gray, i);
    i <<= 1;
  }
  for (i = 1; i < 4; i <<= 1) {
    for (j = i << 1; j < 8; j <<= 1) {
      if (((gray[0] & i) != (start_point & i)) &&
          (gray[0] & j) != (start_point & j))
        gray_swap(gray, i, j);
      else if (((gray[7] & i) != (end_point & i)) &&
               (gray[7] & j) != (end_point & j))
        gray_swap(gray, i, j);
    }
  }
  u8 dirs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (i = 0; i < 8; i++) {
    u8 a = gray[i], b = gray[(i + 1) & 7];
    u8 diff = a ^ b;
    assert(popcount(diff) == 1);
    u8 sign = !(b & diff); // 0 = pos, 1 = neg
    dirs[i] = !!(diff & 1) * DIR_PX + !!(diff & 2) * DIR_PY +
              !!(diff & 4) * DIR_PZ + sign;
    assert(dirs[i] < 6);
  }
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
  assert(is_gray(out_gray));
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
        stk[stk_ptr++] = half_idx_add_dir_2(top, i);
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
    assert(valid_idx(top));
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
        stk[stk_ptr++] = half_idx_add_dir_3(top, dirs[i]);
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
  x += 5;
  x ^= x >> 17;
  x *= 0xed5ad4bbU;
  x ^= x >> 11;
  x *= 0xac4c1b51U;
  x ^= x >> 15;
  x *= 0x31848babU;
  x ^= x >> 14;
  return x;
}

u32 *make_edge_weights(u32 num_edges) {
  u32 *out = malloc(sizeof(u32) * num_edges);
  assert(out);
  for (u32 i = 0; i < num_edges; i++) {
    out[i] = triple32(i);
  }
  return out;
}

#include <unistd.h>

void run_pic(u8 *screen_dirs, u8 *cube_dirs, u32 screen_idx, u32 cube_idx) {
  u32 orig_cube_idx = cube_idx;
  FILE *f = fopen("out.pbm", "w");
  u32 *pix = malloc(sizeof(u32) * x_span() * y_span());
  u8 *out_f = malloc(sizeof(u8) * x_span() * y_span() * 3);
  assert(pix && out_f);
  u32 out_f_ptr = 0;
  fprintf(f, "P6\n%i %i\n255\n", x_span(), y_span());
  do {
    screen_idx = idx_add_dir_2(screen_idx, screen_dirs[screen_idx]);
    cube_idx = idx_add_dir_3(cube_idx, cube_dirs[cube_idx]);
    pix[screen_idx] = cube_idx;
  } while (cube_idx != orig_cube_idx);
  for (u32 i = 0; i < x_span() * y_span(); i++) {
    u32 cube_idx = pix[i];
    u8 r = cb_point_r(cube_idx), g = cb_point_g(cube_idx),
       b = cb_point_b(cube_idx);
    r = (r << (8 - r_bits));
    g = (g << (8 - g_bits));
    b = (b << (8 - b_bits));
    out_f[out_f_ptr++] = r;
    out_f[out_f_ptr++] = g;
    out_f[out_f_ptr++] = b;
  }
  fwrite(out_f, out_f_ptr, 1, f);
  fclose(f);
  free(out_f);
  free(pix);
}

int main(int argc, const char *const *argv) {
  u32 num_screen_edges, num_cube_edges;
  u32 num_screen_nodes = x_span() / 2 * y_span() / 2;
  u32 num_cube_nodes = r_span() / 2 * g_span() / 2 * b_span() / 2;

  printf("making edges...\n");
  edge *screen_edges = make_screen_edges(x_span(), y_span(), &num_screen_edges);
  edge *cube_edges =
      make_cube_edges(r_span(), g_span(), b_span(), &num_cube_edges);
  printf("making edge weights...\n");
  u32 *screen_edge_weights = make_edge_weights(num_screen_edges);
  u32 *cube_edge_weights = make_edge_weights(num_cube_edges);
  printf("kruskinating...\n");
  screen_edges = kruskinate(num_screen_nodes, num_screen_edges, screen_edges,
                            screen_edge_weights);
  cube_edges =
      kruskinate(num_cube_nodes, num_cube_edges, cube_edges, cube_edge_weights);
  printf("mapping edges...\n");
  u8 *screen_edge_dirs =
      map_edges(num_screen_nodes, num_screen_nodes - 1, screen_edges, 2);
  u8 *cube_edge_dirs =
      map_edges(num_cube_nodes, num_cube_nodes - 1, cube_edges, 3);
  printf("reorienting edges...\n");
  screen_edge_dirs = reorient_edges(num_screen_nodes, screen_edge_dirs, 0, 2);
  cube_edge_dirs = reorient_edges(num_cube_nodes, cube_edge_dirs, 0, 3);
  check_mst(screen_edge_dirs, num_screen_nodes, 0, 2);
  check_mst(cube_edge_dirs, num_cube_nodes, 0, 3);
  printf("resolving edges...\n");
  u8 *out_screen_dirs = resolve_edges_2(num_screen_nodes, screen_edge_dirs, 0);
  u8 *out_cube_dirs = resolve_edges_3(num_cube_nodes, cube_edge_dirs, 0);
  run_pic(out_screen_dirs, out_cube_dirs, 0, 0);
}
