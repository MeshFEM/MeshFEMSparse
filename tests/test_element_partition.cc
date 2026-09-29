#include <MeshFEMSparse/ElementPartitionFromND.hh>
#include <array>
#include <functional>
#include <catch2/catch.hpp>

namespace {
using P = MeshFEM::ElementPartitionFromND<>;
using Tri = std::array<int, 3>;
void require(bool v) { if (!v) throw std::runtime_error("Partition test failed"); }
void rejects(const std::function<void()> &f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument &) { caught = true; }
    require(caught);
}
template<class Elements> auto getter(const Elements &e) {
    return [&e](size_t ei) -> const auto & { return e[ei]; };
}
template<class Elements> P build(int nv, const Elements &e, const std::vector<int> &parent,
                                const std::vector<int> &member, int depth) {
    P p(e.size(), getter(e), parent, member, depth);
    p.validate(e.size(), getter(e));
    require(p.numBlockVars() == size_t(nv) && p.numElements() == e.size());
    // Locks mark exactly the vertices shared by multiple partitions.
    std::vector<int> owner(nv, -1);
    std::vector<uint8_t> shared(nv, false);
    for (int partition = 0; partition < p.numPartitions(); ++partition) {
        for (int i = p.partitionOffsets[partition]; i < p.partitionOffsets[partition + 1]; ++i) {
            for (auto v : e[p.elementOrder[i]]) {
                if (owner[v] < 0) owner[v] = partition;
                else if (owner[v] != partition) shared[v] = true;
            }
        }
    }
    require(shared == p.variableNeedsLock);
    return p;
}
TEST_CASE("ND element partition topology", "[nd_partition]") {
    std::vector<Tri> e = {{{0,1,2}}, {{1,0,3}}};
    std::vector<int> parent = {2,2,-1}, member = {2,2,0,1};
    auto p = build(4,e,parent,member,1);
    require(p.numPartitions() == 2 && p.variableNeedsLock == std::vector<uint8_t>({1,1,0,0}));
    require(build(4,e,parent,member,0).numPartitions() == 1);
    require(build(4,e,parent,member,20).numPartitions() == 2); // leaves never separators
    auto e2 = e; e2.push_back({{0,1,4}}); auto m2 = member; m2.push_back(2);
    auto attached = build(5,e2,parent,m2,1);
    require(attached.numPartitions() == 2 && attached.variableNeedsLock == std::vector<uint8_t>({1,1,0,0,0}));
    // Connectivity within a label is irrelevant, including non-manifold stars.
    require(build(5,std::vector<Tri>{{{0,1,2}},{{0,3,4}}},{-1},{0,0,0,0,0},4).numPartitions() == 1);
    require(build(5,std::vector<Tri>{{{0,1,2}},{{1,0,3}},{{0,1,4}}},{-1},{0,0,0,0,0},2).numPartitions() == 1);
    require(build(7,std::vector<Tri>{{{0,1,2}},{{3,4,5}}},{-1,-1},{0,0,0,1,1,1,1},0).numPartitions() == 2);
    auto all = build(3,std::vector<Tri>{{{0,1,2}}},{1,-1},{1,1,1},1);
    require(all.numPartitions() == 1);
    require(build(0,std::vector<Tri>{},{},{},2).numPartitions() == 0);
    require(build(4,std::vector<Tri>{},parent,member,1).numPartitions() == 0);
    auto tet = build(5,std::vector<std::array<int,4>>{{{0,1,2,3}},{{2,1,0,4}}},parent,{2,2,2,0,1},1);
    require(tet.numPartitions() == 2 && std::count(tet.variableNeedsLock.begin(),tet.variableNeedsLock.end(),1) == 3);

    // An unbalanced tree: shallow leaf 0 remains interior; descendants 1 and 2
    // inherit the cutoff component 3. Component 4 alone is a separator at depth 1.
    std::vector<std::vector<size_t>> mixed = {{0,1,2,3,4,5}, {0,6,7,8}, {0}, {}, {1,1}};
    std::vector<int> unbalanced = {4,3,3,4,-1};
    std::vector<int> mixedMember = {4,0,0,0,0,0,1,2,3};
    auto shallow = build(9,mixed,unbalanced,mixedMember,1);
    require(shallow.numPartitions() == 2);
    require(shallow.variableNeedsLock == std::vector<uint8_t>({1,0,0,0,0,0,0,0,0}));
    require(shallow.partitionOffsets == std::vector<int>({0,3,5}));
    require(shallow.elementOrder == std::vector<int>({2,0,4,3,1}));
    // Splitting one level deeper makes the second stencil cross interiors.
    rejects([&] { build(9,mixed,unbalanced,mixedMember,2); });
    // A six-node shell stencil and a quad need no facet/topology special cases.
    auto unused = build(3,std::vector<Tri>{{{0,1,2}}},{2,2,-1},{0,0,0},1);
    require(unused.numPartitions() == 1 && unused.variableNeedsLock == std::vector<uint8_t>(3,0));
    require(build(0,std::vector<std::vector<int>>{{},{}},{},{},1).numPartitions() == 1);

    rejects([&] { build(4,e,{1,0},member,1); });
    rejects([&] { build(4,e,{2,9,-1},member,1); });
    rejects([&] { build(4,e,parent,{2,2,0,9},1); });
    rejects([&] { build(4,e,parent,{0,1,0,1},1); });
    rejects([&] { build(4,e,parent,member,-1); });
    rejects([&] { build(4,std::vector<Tri>{{{-1,0,1}}},parent,member,1); });
    rejects([&] { build(4,std::vector<std::array<size_t,2>>{{{0,size_t(-1)}}},parent,member,1); });
    // Sorting is automatic, uses current variable indices, and ignores stencil
    // orientation. Equal keys retain original element-ID order; prefixes and
    // empty stencils sort correctly for variable-size elements.
    std::vector<std::vector<int>> stencils = {{2,1},{1,2},{1},{},{0,2},{2,0},{1,0,2}};
    auto original = stencils;
    auto sorted = build(3,stencils,{-1},{0,0,0},0);
    require(sorted.elementOrder == std::vector<int>({3,6,4,5,2,0,1}));
    require(stencils == original);
    auto bad = p; bad.variableNeedsLock[0] = false;
    rejects([&] { bad.validate(e.size(),getter(e)); });
    bad = p; bad.partitionOffsets[1] = 100;
    rejects([&] { bad.validate(e.size(),getter(e)); });
    bad = p; bad.partitionOffsets[1] = -1;
    rejects([&] { bad.validate(e.size(),getter(e)); });
    bad = p; bad.elementOrder[1] = -1;
    rejects([&] { bad.validate(e.size(),getter(e)); });
    bad = p; bad.elementOrder[1] = bad.elementOrder[0];
    rejects([&] { bad.validate(e.size(),getter(e)); });
}

} // namespace
