//! PyO3 bindings — a thin shell over `belief_graph`.
//!
//! Deliberately thin: every algorithm lives in pure Rust so it can be tested
//! with `cargo test`, and this file only translates types across the boundary.

use pyo3::prelude::*;

pub mod belief_graph;

use belief_graph::{BeliefGraph, Kind};

const BELIEF: &str = "belief";
const CONCEPT: &str = "concept";

fn kind_from(name: &str) -> PyResult<Kind> {
    match name {
        BELIEF => Ok(Kind::Belief),
        CONCEPT => Ok(Kind::Concept),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown node kind {other:?}; expected 'belief' or 'concept'"
        ))),
    }
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Belief => BELIEF,
        Kind::Concept => CONCEPT,
    }
}

/// Native belief graph. Mirrors the Python `KnowledgeGraph` interface.
#[pyclass(name = "BeliefGraph")]
struct PyBeliefGraph {
    inner: BeliefGraph,
}

#[pymethods]
impl PyBeliefGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: BeliefGraph::new(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn add_belief(&mut self, id: &str, confidence: f64, connections: Vec<String>) {
        self.inner.add_belief(id, confidence, &connections);
    }

    fn neighbours(&self, kind: &str, key: &str) -> PyResult<Vec<String>> {
        Ok(self.inner.neighbours(kind_from(kind)?, key))
    }

    #[pyo3(signature = (kind, key, max_hops=3))]
    fn walk(
        &self,
        kind: &str,
        key: &str,
        max_hops: usize,
    ) -> PyResult<Vec<(String, String, usize)>> {
        Ok(self
            .inner
            .walk(kind_from(kind)?, key, max_hops)
            .into_iter()
            .map(|(k, key, depth)| (kind_name(k).to_string(), key, depth))
            .collect())
    }

    #[pyo3(signature = (belief_id, max_hops=2))]
    fn related(&self, belief_id: &str, max_hops: usize) -> Vec<String> {
        self.inner.related(belief_id, max_hops)
    }

    #[pyo3(signature = (from_kind, from_key, to_kind, to_key, max_hops=6))]
    fn path(
        &self,
        from_kind: &str,
        from_key: &str,
        to_kind: &str,
        to_key: &str,
        max_hops: usize,
    ) -> PyResult<Option<Vec<(String, String)>>> {
        let from = (kind_from(from_kind)?, from_key);
        let to = (kind_from(to_kind)?, to_key);
        Ok(self.inner.path(from, to, max_hops).map(|chain| {
            chain
                .into_iter()
                .map(|(k, key)| (kind_name(k).to_string(), key))
                .collect()
        }))
    }

    fn path_strength(&self, path: Vec<(String, String)>) -> PyResult<f64> {
        let chain: PyResult<Vec<(Kind, String)>> = path
            .into_iter()
            .map(|(kind, key)| Ok((kind_from(&kind)?, key)))
            .collect();
        Ok(self.inner.path_strength(&chain?))
    }
}

#[pymodule]
fn cme_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBeliefGraph>()?;
    m.add("__doc__", "Native belief graph traversal for CME.")?;
    Ok(())
}
