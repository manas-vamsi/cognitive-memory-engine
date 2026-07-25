//! Bipartite belief/concept graph and the traversals over it.
//!
//! Pure Rust, no Python types — so `cargo test` exercises the algorithm without
//! linking against libpython, and the PyO3 layer in `lib.rs` stays a thin shell.
//!
//! This mirrors the Python `KnowledgeGraph`, which is the reference
//! implementation. Same results, same edge cases; only faster.

use std::collections::{HashMap, HashSet, VecDeque};

/// Nodes are interned to `usize` so adjacency is index lookups, not string
/// hashing on every hop — the whole reason this lives in Rust.
pub type NodeId = usize;

/// `Belief` is declared first so `Ord` puts it before `Concept`, matching the
/// Python side where the kind is the string "belief" or "concept" and sorts
/// lexicographically. Both implementations must break ties identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Belief,
    Concept,
}

#[derive(Debug, Default)]
pub struct BeliefGraph {
    edges: Vec<HashSet<NodeId>>,
    kinds: Vec<Kind>,
    keys: Vec<String>,
    index: HashMap<(Kind, String), NodeId>,
    confidence: HashMap<NodeId, f64>,
}

impl BeliefGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    fn intern(&mut self, kind: Kind, key: &str) -> NodeId {
        if let Some(&id) = self.index.get(&(kind, key.to_string())) {
            return id;
        }
        let id = self.edges.len();
        self.edges.push(HashSet::new());
        self.kinds.push(kind);
        self.keys.push(key.to_string());
        self.index.insert((kind, key.to_string()), id);
        id
    }

    fn lookup(&self, kind: Kind, key: &str) -> Option<NodeId> {
        self.index.get(&(kind, key.to_string())).copied()
    }

    fn link(&mut self, a: NodeId, b: NodeId) {
        self.edges[a].insert(b);
        self.edges[b].insert(a);
    }

    /// Neighbours in a fixed (kind, key) order.
    ///
    /// `HashSet` iteration order is unspecified and varies between runs, so an
    /// unsorted BFS can return a different — equally short — path each time.
    /// For an engine that presents its path as the reasoning trace, an
    /// explanation that changes run to run is a defect. Sorting costs
    /// O(k log k) per expansion and buys a reproducible answer that matches the
    /// Python reference exactly.
    fn ordered_neighbours(&self, node: NodeId) -> Vec<NodeId> {
        let mut ns: Vec<NodeId> = self.edges[node].iter().copied().collect();
        ns.sort_by(|&a, &b| (self.kinds[a], &self.keys[a]).cmp(&(self.kinds[b], &self.keys[b])));
        ns
    }

    /// Add a belief and wire it to every concept it mentions.
    pub fn add_belief(&mut self, id: &str, confidence: f64, connections: &[String]) {
        let node = self.intern(Kind::Belief, id);
        self.confidence.insert(node, confidence);
        for label in connections {
            if label.is_empty() {
                continue;
            }
            let concept = self.intern(Kind::Concept, label);
            self.link(node, concept);
        }
    }

    pub fn neighbours(&self, kind: Kind, key: &str) -> Vec<String> {
        match self.lookup(kind, key) {
            None => Vec::new(),
            Some(node) => self.edges[node]
                .iter()
                .map(|&n| self.keys[n].clone())
                .collect(),
        }
    }

    /// Breadth-first walk outward, yielding (kind, key, hop distance).
    pub fn walk(&self, kind: Kind, key: &str, max_hops: usize) -> Vec<(Kind, String, usize)> {
        let Some(start) = self.lookup(kind, key) else {
            return Vec::new();
        };
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([(start, 0usize)]);
        let mut out = Vec::new();
        while let Some((node, depth)) = queue.pop_front() {
            out.push((self.kinds[node], self.keys[node].clone(), depth));
            if depth >= max_hops {
                continue;
            }
            for next in self.ordered_neighbours(node) {
                if seen.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
        }
        out
    }

    /// Other beliefs reachable within `max_hops`, nearest first then strongest.
    pub fn related(&self, belief_id: &str, max_hops: usize) -> Vec<String> {
        let mut found: Vec<(usize, f64, String)> = self
            .walk(Kind::Belief, belief_id, max_hops)
            .into_iter()
            .filter(|(kind, key, _)| *kind == Kind::Belief && key != belief_id)
            .map(|(_, key, depth)| {
                let conf = self
                    .lookup(Kind::Belief, &key)
                    .and_then(|n| self.confidence.get(&n).copied())
                    .unwrap_or(0.0);
                (depth, -conf, key)
            })
            .collect();
        // Nearest first; ties broken by strongest. Confidence is negated above
        // so a plain ascending sort puts the strongest first.
        found.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.2.cmp(&b.2))
        });
        found.into_iter().map(|(_, _, key)| key).collect()
    }

    /// Shortest chain between two nodes as (kind, key) pairs, if one exists.
    pub fn path(
        &self,
        from: (Kind, &str),
        to: (Kind, &str),
        max_hops: usize,
    ) -> Option<Vec<(Kind, String)>> {
        let start = self.lookup(from.0, from.1)?;
        let goal = self.lookup(to.0, to.1)?;
        if start == goal {
            return Some(vec![(self.kinds[start], self.keys[start].clone())]);
        }
        let mut came_from: HashMap<NodeId, NodeId> = HashMap::from([(start, start)]);
        let mut queue = VecDeque::from([(start, 0usize)]);
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }
            for next in self.ordered_neighbours(node) {
                if came_from.contains_key(&next) {
                    continue;
                }
                came_from.insert(next, node);
                if next == goal {
                    return Some(self.trace(&came_from, start, goal));
                }
                queue.push_back((next, depth + 1));
            }
        }
        None
    }

    fn trace(
        &self,
        came_from: &HashMap<NodeId, NodeId>,
        start: NodeId,
        goal: NodeId,
    ) -> Vec<(Kind, String)> {
        let mut chain = vec![goal];
        while *chain.last().unwrap() != start {
            chain.push(came_from[chain.last().unwrap()]);
        }
        chain.reverse();
        chain
            .into_iter()
            .map(|n| (self.kinds[n], self.keys[n].clone()))
            .collect()
    }

    /// Confidence of a chain: the product of the beliefs it crosses.
    pub fn path_strength(&self, path: &[(Kind, String)]) -> f64 {
        path.iter()
            .filter(|(kind, _)| *kind == Kind::Belief)
            .filter_map(|(_, key)| {
                self.lookup(Kind::Belief, key)
                    .and_then(|n| self.confidence.get(&n).copied())
            })
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> BeliefGraph {
        let mut g = BeliefGraph::new();
        g.add_belief(
            "qubits",
            0.9,
            &["quantum computing".into(), "qubits".into()],
        );
        g.add_belief("superpos", 0.8, &["qubits".into()]);
        g.add_belief("island", 0.95, &["rust".into()]);
        g
    }

    #[test]
    fn beliefs_sharing_a_concept_are_related() {
        let g = fixture();
        assert_eq!(g.related("qubits", 2), vec!["superpos".to_string()]);
        assert!(g.related("island", 2).is_empty());
    }

    #[test]
    fn path_runs_through_the_shared_concept() {
        let g = fixture();
        let path = g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "superpos"), 6)
            .expect("reachable");
        assert_eq!(
            path,
            vec![
                (Kind::Belief, "qubits".to_string()),
                (Kind::Concept, "qubits".to_string()),
                (Kind::Belief, "superpos".to_string()),
            ]
        );
    }

    #[test]
    fn belief_and_concept_keys_do_not_collide() {
        // "qubits" is both a belief id and a concept label in the fixture.
        let g = fixture();
        assert_eq!(g.neighbours(Kind::Belief, "qubits").len(), 2);
        assert_eq!(g.neighbours(Kind::Concept, "qubits").len(), 2);
    }

    #[test]
    fn unreachable_and_unknown_nodes_return_none() {
        let g = fixture();
        assert!(g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "island"), 6)
            .is_none());
        assert!(g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "ghost"), 6)
            .is_none());
    }

    #[test]
    fn max_hops_bounds_the_search() {
        let g = fixture();
        assert!(g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "superpos"), 1)
            .is_none());
        assert!(g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "superpos"), 2)
            .is_some());
    }

    #[test]
    fn path_strength_multiplies_belief_confidence() {
        let g = fixture();
        let path = g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "superpos"), 6)
            .unwrap();
        assert!((g.path_strength(&path) - 0.72).abs() < 1e-9);
    }

    #[test]
    fn re_adding_a_belief_updates_rather_than_duplicates() {
        let mut g = fixture();
        let before = g.len();
        g.add_belief("qubits", 0.5, &["qubits".into()]);
        assert_eq!(g.len(), before);
        let path = g
            .path((Kind::Belief, "qubits"), (Kind::Belief, "superpos"), 6)
            .unwrap();
        assert!((g.path_strength(&path) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn a_belief_with_no_connections_is_still_a_node() {
        let mut g = BeliefGraph::new();
        g.add_belief("lonely", 0.5, &[]);
        assert_eq!(g.len(), 1);
        assert!(g.neighbours(Kind::Belief, "lonely").is_empty());
        assert!(g.related("lonely", 2).is_empty());
    }

    #[test]
    fn walking_an_unknown_node_yields_nothing() {
        let g = fixture();
        assert!(g.walk(Kind::Belief, "ghost", 3).is_empty());
    }
}
