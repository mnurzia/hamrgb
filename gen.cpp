#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

struct Edge {
public:
  unsigned int start;
  unsigned int end;

  static unsigned int rgb2idx(unsigned int r, unsigned int g, unsigned int b) {
    return r << 16 | g << 8 | b;
  }

  static unsigned int xy2idx(unsigned int x, unsigned int y) {
    return y * 4096 + x;
  }

  static std::pair<unsigned int, unsigned int> idx2xy(unsigned int idx) {
    return std::make_pair(idx % 4096, idx / 4096);
  }

public:
  Edge(unsigned int sr, unsigned int sg, unsigned int sb, unsigned int dr,
       unsigned int dg, unsigned int db) {
    start = rgb2idx(sr, sg, sb);
    end = rgb2idx(dr, dg, db);
  }

  Edge(unsigned int sx, unsigned int sy, unsigned int dx, unsigned int dy) {
    start = xy2idx(sx, sy);
    end = xy2idx(dx, dy);
  }
};

class ColorGen {
  using epair = std::pair<Edge, unsigned int>;

  std::vector<epair> edges;
  std::vector<Edge> treeEdges;
  std::vector<Edge> cycEdges;
  std::unordered_map<unsigned int, unsigned int> repSet;
  std::unordered_map<unsigned int, unsigned int> repSetSize;
  std::random_device rd;
  std::mt19937 mt;
  std::uniform_int_distribution<unsigned int> distr;

public:
  ColorGen() : mt(rd()), distr(0, 4096 * 4096 - 1) {
    edges.reserve(4096 * 4096);
    treeEdges.reserve(4096 * 4096);
    cycEdges.reserve(4096 * 4096);
  }

private:
  void genEdges() {
    for (int y = 0; y < 4096; y += 2) {
      for (int x = 0; x < 4096; x += 2) {
        if (x != 4096 - 2) {
          edges.emplace_back(Edge(x, y, x + 2, y), distr(mt));
        }
        if (y != 4096 - 2) {
          edges.emplace_back(Edge(x, y, x, y + 2), distr(mt));
        }
        unsigned int idx = Edge::xy2idx(x, y);
        repSet.emplace(idx, idx);
        repSetSize.emplace(idx, 0);
      }
    }
  }

  void sortEdges() {
    std::sort(
        edges.begin(), edges.end(),
        [](std::pair<Edge, unsigned int> a, std::pair<Edge, unsigned int> b) {
          return a.second < b.second;
        });
  }

  unsigned int getRepSet(unsigned int idx) {
    while (repSet[idx] != idx) {
      idx = repSet[idx];
    }
    return idx;
  }

  unsigned int repSetIters(unsigned int idx) {
    unsigned int it = 0;
    while (repSet[idx] != idx) {
      idx = repSet[idx];
      it++;
    }
    return it;
  }

  bool isCompleteTree(unsigned int numNodes, unsigned int numEdges) {
    return numEdges == numNodes - 1;
  }

  void runKruskal() {
    unsigned int popIdx = edges.size();
    while (!isCompleteTree(2048 * 2048, treeEdges.size())) {
      Edge nextEdge = edges[--popIdx].first;
      unsigned int setX = getRepSet(nextEdge.start);
      unsigned int setY = getRepSet(nextEdge.end);
      if (setX != setY) {
        unsigned int szX = repSetSize[setX];
        unsigned int szY = repSetSize[setY];
        if (szX < szY) {
          std::swap(szX, szY);
          std::swap(setX, setY);
        }
        treeEdges.push_back(nextEdge);
        repSet[setY] = setX;
        if (repSetSize[setX] == repSetSize[setY]) {
          repSetSize[setX] = szX + 1;
        }
      }
      if (popIdx % 10000 == 0) {
        std::cout << popIdx << " " << treeEdges.size() << std::endl;
      }
    }
  }

  void genCycEdges() {
    for (int y = 0; y < 4096; y += 2) {
      for (int x = 0; x < 4096; x += 2) {
        cycEdges.emplace_back(x, y, x + 1, y);
        cycEdges.emplace_back(x + 1, y, x + 1, y + 1);
      }
      for (int x = 0; x < 4096; x += 2) {
        cycEdges.emplace_back(x, y + 1, x, y);
        cycEdges.emplace_back(x + 1, y + 1, x, y + 1);
      }
    }
  }

  void joinCycEdges() {
    for (Edge e : treeEdges) {
      auto sxy = Edge::idx2xy(e.start);
      auto exy = Edge::idx2xy(e.end);
      auto sx = sxy.first;
      auto sy = sxy.second;
      auto ex = exy.first;
      auto ey = exy.second;
      if (sx < ex) {
        // horz edge, join right
        cycEdges[Edge::xy2idx(sx + 1, sy)] = Edge(sx + 1, sy, sx + 2, sy);
        cycEdges[Edge::xy2idx(sx + 2, sy + 1)] =
            Edge(sx + 2, sy + 1, sx + 1, sy + 1);
      } else if (sy < ey) {
        // vert edge, join down
        cycEdges[Edge::xy2idx(sx + 1, sy + 1)] =
            Edge(sx + 1, sy + 1, sx + 1, sy + 2);
        cycEdges[Edge::xy2idx(sx, sy + 2)] = Edge(sx, sy + 2, sx, sy + 1);
      } else {
        assert(0);
      }
    }
  }

  void writePic() {
    auto sx = distr(mt) % 4096;
    auto sy = distr(mt) % 4096;
    auto sxy = Edge::xy2idx(sx, sy);
    auto xy = sxy;
    std::cout << sx << "," << sy << std::endl;
    unsigned int color = 0;
    std::vector<unsigned int> pixels;
    pixels.reserve(4096 * 4096);
    for (int i = 0; i < 4096 * 4096; i++) {
      pixels.push_back(0);
    }
    do {
      pixels[xy] = color++;
      xy = cycEdges[xy].end;
    } while (xy != sxy);
    std::ofstream out("out.pbm", std::ios::binary);
    std::string hdr("P6\n4096 4096\n255\n");
    out.write(hdr.c_str(), hdr.size());
    for (unsigned int i = 0; i < 4096 * 4096; i++) {
      unsigned int pix = pixels[i];
      char r, g, b;
      r = pix & 255;
      g = (pix >> 8) & 255;
      b = (pix >> 16) & 255;
      out.write(&r, 1);
      out.write(&g, 1);
      out.write(&b, 1);
    }
    out.close();
  }

public:
  void gen() {
    genEdges();
    sortEdges();
    runKruskal();
    genCycEdges();
    joinCycEdges();
    writePic();
  }
};

int main() {
  ColorGen cg;
  cg.gen();
  return 0;
}
