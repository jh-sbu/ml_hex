use std::sync::Arc;

use hex_agents::Agent;

pub enum PlayerConfig {
    Human,
    Agent(Arc<dyn Agent>),
}

impl PlayerConfig {
    pub fn agent<A: Agent + 'static>(a: A) -> Self {
        Self::Agent(Arc::new(a))
    }

    pub fn is_human(&self) -> bool {
        matches!(self, Self::Human)
    }
}

pub struct TuiConfig {
    pub red: PlayerConfig,
    pub blue: PlayerConfig,
    pub swap: bool,
}

impl TuiConfig {
    pub fn new(red: PlayerConfig, blue: PlayerConfig, swap: bool) -> Self {
        Self { red, blue, swap }
    }
}
