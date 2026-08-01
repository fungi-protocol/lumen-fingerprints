mod consumer;

use crate::consumer::SurveyConsumer;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};

use bitcoin::Network;
use floresta_chain::{
    AssumeValidArg, BlockchainInterface, ChainParams, ChainState, FlatChainStore,
    FlatChainStoreConfig,
};
use floresta_mempool::Mempool;
use floresta_wire::UtreexoNodeConfig;
use floresta_wire::address_man::{AddressMan, ReachableNetworks};
use floresta_wire::node::UtreexoNode;
use floresta_wire::node::running_ctx::RunningNode;
use lumen_primitives::traits::{BlockSource, SourcedBlock};

const MEMPOOL_SIZE: usize = 1_000_000;

#[derive(Debug, Clone)]
pub struct FlorestaConfig {
    pub datadir: PathBuf,
    pub start_height: u32,
    pub stop_height: Option<u32>,
    pub channel_capacity: usize,
    pub fixed_peers: Vec<String>,
}

impl FlorestaConfig {
    pub fn new(datadir: PathBuf, start_height: u32) -> Self {
        Self {
            datadir,
            start_height,
            stop_height: None,
            channel_capacity: 64,
            fixed_peers: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum FlorestaSourceError {
    Chain(String),
    Node(String),
    Disconnected,
}

impl std::fmt::Display for FlorestaSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain(msg) => write!(f, "chainstate error: {msg}"),
            Self::Node(msg) => write!(f, "node error: {msg}"),
            Self::Disconnected => write!(f, "node stopped before the window completed"),
        }
    }
}

impl std::error::Error for FlorestaSourceError {}

/// Streams mainnet blocks with utreexo-resolved prevouts.
///
/// Owns a tokio runtime running the Floresta node; dropping it stops the node.
pub struct FlorestaSource {
    receiver: Receiver<SourcedBlock>,
    stop_height: Option<u32>,
    last_height: Option<u32>,
    kill: Arc<tokio::sync::RwLock<bool>>,
    _runtime: tokio::runtime::Runtime,
}

impl FlorestaSource {
    pub fn start(config: FlorestaConfig) -> Result<Self, FlorestaSourceError> {
        let runtime =
            tokio::runtime::Runtime::new().map_err(|e| FlorestaSourceError::Node(e.to_string()))?;

        // `FlatChainStoreConfig::new` is generic over `impl AsRef<Path>`; passing the
        // `&str` directly (rather than `.into()`) avoids an ambiguous-type-inference
        // compile error (confirmed in the Task 1 spike, see
        // docs/floresta-spike-results.md, correction #2).
        let store_config = FlatChainStoreConfig::new(
            config
                .datadir
                .to_str()
                .ok_or_else(|| FlorestaSourceError::Chain("non-utf8 datadir".into()))?,
        );
        let store = FlatChainStore::new(store_config)
            .map_err(|e| FlorestaSourceError::Chain(format!("{e:?}")))?;
        // `Network` here is `bitcoin::Network`, not a `floresta_chain` type (Task 1
        // spike correction #1: `floresta_chain::Network` does not exist).
        let chain = Arc::new(
            ChainState::open(store, Network::Bitcoin, AssumeValidArg::Hardcoded)
                .map_err(|e| FlorestaSourceError::Chain(format!("{e:?}")))?,
        );

        let (sender, receiver) = sync_channel(config.channel_capacity);
        chain.subscribe(Arc::new(SurveyConsumer::new(
            sender,
            config.start_height,
            config.stop_height,
        )));

        let node_config = UtreexoNodeConfig {
            network: Network::Bitcoin,
            assume_utreexo: Some(ChainParams::get_assume_utreexo(Network::Bitcoin)),
            // `false`, not `true`: florestad's own reference config always disables
            // PoW fraud proofs when `assume_utreexo` is set, treating assume-utreexo
            // as the sole fast-IBD mechanism (Task 1 spike correction #3). The brief's
            // sample code had `true`; that combination is untested upstream.
            pow_fraud_proofs: false,
            backfill: false,
            fixed_peers: config.fixed_peers.clone(),
            datadir: config.datadir.clone(),
            ..Default::default()
        };

        let kill = Arc::new(tokio::sync::RwLock::new(false));
        let node: UtreexoNode<Arc<ChainState<FlatChainStore>>, RunningNode> = UtreexoNode::new(
            node_config,
            chain.clone(),
            Arc::new(tokio::sync::Mutex::new(Mempool::new(MEMPOOL_SIZE))),
            None,
            kill.clone(),
            AddressMan::new(None, &ReachableNetworks::SUPPORTED),
        )
        .map_err(|e| FlorestaSourceError::Node(format!("{e:?}")))?;

        let (stop_tx, _stop_rx) = tokio::sync::oneshot::channel();
        runtime.spawn(node.run(stop_tx));

        Ok(Self {
            receiver,
            stop_height: config.stop_height,
            last_height: None,
            kill,
            _runtime: runtime,
        })
    }
}

/// The chain tip and assume-utreexo floor of a Floresta `FlatChainStore`, as reported by
/// `sync_tip`. `tip` is the best block height once headers are synced; `floor` is the
/// assume-utreexo height, the lowest block Floresta can serve. Neither is a hardcoded
/// constant: `tip` is read from the store, `floor` from the network's chain params.
pub struct TipInfo {
    pub tip: u32,
    pub floor: u32,
    pub in_ibd: bool,
}

/// Start a Floresta node against `datadir`, sync the header chain from the network, and
/// report the live tip once the header height stabilizes (no growth for a settling window)
/// or a timeout elapses. This reflects the *current* network tip, not whatever a store
/// happens to already hold. Needs peers and can take a few minutes on a fresh datadir.
/// The node is stopped before returning. Progress is logged to stderr.
pub fn sync_tip(
    datadir: &std::path::Path,
    fixed_peers: Vec<String>,
) -> Result<TipInfo, FlorestaSourceError> {
    use std::time::{Duration, Instant};

    let runtime =
        tokio::runtime::Runtime::new().map_err(|e| FlorestaSourceError::Node(e.to_string()))?;
    let store_config = FlatChainStoreConfig::new(
        datadir
            .to_str()
            .ok_or_else(|| FlorestaSourceError::Chain("non-utf8 datadir".into()))?,
    );
    let store = FlatChainStore::new(store_config)
        .map_err(|e| FlorestaSourceError::Chain(format!("{e:?}")))?;
    let chain = Arc::new(
        ChainState::open(store, Network::Bitcoin, AssumeValidArg::Hardcoded)
            .map_err(|e| FlorestaSourceError::Chain(format!("{e:?}")))?,
    );

    let node_config = UtreexoNodeConfig {
        network: Network::Bitcoin,
        assume_utreexo: Some(ChainParams::get_assume_utreexo(Network::Bitcoin)),
        pow_fraud_proofs: false,
        backfill: false,
        fixed_peers,
        datadir: datadir.to_path_buf(),
        ..Default::default()
    };
    let kill = Arc::new(tokio::sync::RwLock::new(false));
    let node: UtreexoNode<Arc<ChainState<FlatChainStore>>, RunningNode> = UtreexoNode::new(
        node_config,
        chain.clone(),
        Arc::new(tokio::sync::Mutex::new(Mempool::new(MEMPOOL_SIZE))),
        None,
        kill.clone(),
        AddressMan::new(None, &ReachableNetworks::SUPPORTED),
    )
    .map_err(|e| FlorestaSourceError::Node(format!("{e:?}")))?;
    let (stop_tx, _stop_rx) = tokio::sync::oneshot::channel();
    runtime.spawn(node.run(stop_tx));

    let floor = ChainParams::get_assume_utreexo(Network::Bitcoin).height;
    // Poll the header tip: `get_best_block` tracks the best header (climbs during header
    // download), while validation lags (`is_in_ibd` stays true), so "height stopped
    // growing" — not `is_in_ibd` — is the header-sync-complete signal.
    let settle = Duration::from_secs(20);
    let timeout = Duration::from_secs(600);
    let mut last = 0u32;
    let mut stable_since = Instant::now();
    let started = Instant::now();
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let (height, _) = chain
            .get_best_block()
            .map_err(|e| FlorestaSourceError::Chain(format!("{e:?}")))?;
        if height != last {
            eprintln!("header sync: height {height}");
            last = height;
            stable_since = Instant::now();
        }
        let settled = height > floor && stable_since.elapsed() >= settle;
        if settled || started.elapsed() >= timeout {
            runtime.block_on(async { *kill.write().await = true });
            return Ok(TipInfo {
                tip: height,
                floor,
                in_ibd: chain.is_in_ibd(),
            });
        }
    }
}

impl BlockSource for FlorestaSource {
    type Error = FlorestaSourceError;

    fn next_block(&mut self) -> Result<Option<SourcedBlock>, Self::Error> {
        if let (Some(stop), Some(last)) = (self.stop_height, self.last_height)
            && last >= stop
        {
            return Ok(None);
        }

        match self.receiver.recv() {
            Ok(block) => {
                self.last_height = Some(block.height);
                Ok(Some(block))
            }
            Err(_) => Err(FlorestaSourceError::Disconnected),
        }
    }
}

impl Drop for FlorestaSource {
    fn drop(&mut self) {
        // Best-effort node shutdown; the runtime is dropped right after.
        //
        // Deliberately `self._runtime.handle()`, not `tokio::runtime::Handle::try_current()`:
        // `start` is called from, and this `Drop` runs on, an ordinary synchronous caller
        // thread that is never itself inside the owned runtime, so `try_current()` would
        // fail here and the kill flag would silently never be set. `Runtime::handle()`
        // reaches the owned runtime directly regardless of the calling thread's context.
        let kill = self.kill.clone();
        self._runtime
            .handle()
            .spawn(async move { *kill.write().await = true });
    }
}
