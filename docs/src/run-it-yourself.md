# Run a scan

<div id="runcard" hidden style="border:1px solid #274034;border-radius:10px;padding:16px;margin:8px 0 20px;background:#0f1512;color:#c6e6d0;font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,Consolas,monospace">
  <p id="datasrc" style="margin:0 0 10px;color:#6f8a78">Pull fresh data from Bitcoin — scanned locally with Floresta — into the Explorer.</p>
  <button id="run" style="font-size:15px;padding:8px 16px;cursor:pointer;font-family:inherit;background:#12341f;color:#c6e6d0;border:1px solid #274034;border-radius:6px">Run a scan</button>
  <button id="stop" hidden style="font-size:15px;padding:8px 16px;cursor:pointer;font-family:inherit;background:#12341f;color:#c6e6d0;border:1px solid #274034;border-radius:6px">Stop</button>
  <p id="diskest" style="margin:8px 0 0;color:#4f6a58;font-size:12px">≈ 5 GB on disk — chain headers (~2 GB) + scan output (~3 GB).</p>
  <div id="scanstat" style="margin-top:10px;color:#6f8a78"></div>
  <div id="scantrack" hidden style="height:14px;background:#17231c;border-radius:5px;overflow:hidden;margin:8px 0"><div id="scanbar" style="height:100%;width:0;background:#3fb950"></div></div>
  <pre id="scanlog" hidden style="max-height:220px;overflow:auto;background:#0a0e0c;color:#c6e6d0;padding:10px 12px;border-radius:8px;font-size:12px;white-space:pre-wrap;margin:10px 0 0;border:1px solid #1c2620"></pre>
  <p id="scandone" hidden style="margin:10px 0 0"><a href="explorer.html" style="color:#3fb950">Open the Fingerprint Explorer</a> to see the data on disk.</p>
</div>

Run a scan reads block-range transaction construction via Floresta (with utreexo-resolved prevouts); that's the only supported input for this Explorer.

Prefer the terminal? Run <code id="termcmd">nix run .#scan</code> — it scans and opens the dashboard for you.

Already have the data? Run `nix run .#serve` to open the Explorer without scanning — or `nix run .#emit -- <report.json>` to load a scan you already ran.

<p id="nobackend" hidden>To run a scan, serve this locally with <code>nix run .#dashboard</code> — then a <b>Run a scan</b> button appears here and fills the <a href="explorer.html">Explorer</a> with fresh numbers. (A hosted page can only show data, not scan it.)</p>

<script>
(function(){
  var $ = function(id){ return document.getElementById(id); };
  var es = null, stopped = false;
  function ui(running){
    $("run").hidden = running; $("stop").hidden = !running;
    $("scantrack").hidden = !running; $("scanlog").hidden = !running;
    if(running) $("scandone").hidden = true;
  }
  function attach(mode){
    es = new EventSource("api/scan?mode="+mode);
    es.addEventListener("log", function(e){ var l=$("scanlog"); l.textContent += JSON.parse(e.data).line+"\n"; l.scrollTop=l.scrollHeight; });
    es.addEventListener("progress", function(e){ var p=JSON.parse(e.data);
      $("scanstat").textContent = (p.pct!=null) ? ("height "+p.height+" · epoch "+p.epochs+" · "+p.pct+"%") : ("height "+p.height+" · epoch "+p.epochs);
      $("scanbar").style.width = (p.pct!=null?p.pct:0)+"%"; });
    es.addEventListener("done", function(){ if(es){es.close();es=null;} ui(false);
      $("scanstat").textContent = stopped ? "Scan stopped." : "Scan complete."; $("scandone").hidden=false; });
    // A transient drop (tab backgrounded, brief network hiccup) is NOT completion --
    // only the `done` event ends a scan. EventSource reconnects on its own, and
    // /api/scan re-attaches and replays the buffered log on reconnect, so there is
    // nothing to finalize here.
    es.onerror = function(){ $("scanstat").textContent = "Reconnecting…"; };
  }
  fetch("api/status",{cache:"no-store"}).then(function(r){return r.json();}).then(function(st){
    if(st && st.backend){
      $("runcard").hidden = false;   // local backend -> show the button
      $("termcmd").textContent = st.repo_name ? ("cd "+st.repo_name+" && nix run .#scan") : "nix run .#scan";
      if(st.running){
        // A scan is already running -- started before this page load, or from
        // another tab/window -- so re-attach immediately instead of showing the
        // idle Run button. This is what lets returning to the page resume
        // showing live logs/progress instead of losing them.
        stopped = false; $("scanlog").textContent = ""; ui(true);
        attach("resume");
      }
    } else {
      $("nobackend").hidden = false;  // hosted, no backend -> show the how-to
    }
  }).catch(function(){ $("nobackend").hidden = false; });
  $("run").onclick = async function(){
    var st = {}; try{ st = await (await fetch("api/status",{cache:"no-store"})).json(); }catch(e){}
    var mode;
    if(st.resume_height){
      mode = confirm("A previous scan reached height "+st.resume_height+".\n\nResuming is faster, but the first epoch after a resume can be silently under-counted (assume-utreexo), so a resumed run is NOT a faithful measurement — restart from the floor for that.\n\nOK = resume · Cancel = restart from the floor") ? "resume" : "fresh";
    } else {
      if(!confirm("Run a scan? It pulls fresh data from the Bitcoin network and can take a while.")) return;
      mode = "fresh";
    }
    stopped = false; $("scanlog").textContent = ""; ui(true);
    attach(mode);
  };
  $("stop").onclick = function(){ stopped = true; $("scanstat").textContent="Stopping…"; fetch("api/stop",{method:"POST"}); };
})();
</script>
