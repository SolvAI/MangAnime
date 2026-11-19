// Blind A/B rating page: synchronised playback + keyboard shortcuts. No dependencies.
(function () {
  const vL = document.getElementById('vL'), vR = document.getElementById('vR');
  if (!vL || !vR) return;
  const vids = [vL, vR];
  const form = document.getElementById('rateform');
  const crits = Array.from(document.querySelectorAll('.crit'));
  const submitBtn = document.getElementById('submit');
  const seek = document.getElementById('seek');
  const playBtn = document.getElementById('play');
  const loopBox = document.getElementById('loop');
  const timeLbl = document.getElementById('time');
  const t0 = performance.now();
  let active = 0, wantPlay = true, audio = 'off';

  const dur = () => Math.max(vL.duration || 0, vR.duration || 0);
  const ended = v => v.ended || (v.duration && v.currentTime >= v.duration - 0.03);

  function setActive(i) {
    active = Math.max(0, Math.min(crits.length - 1, i));
    crits.forEach((c, k) => c.classList.toggle('active', k === active));
  }
  function complete() { return crits.every(c => c.querySelector('input:checked')); }
  function refresh() {
    submitBtn.disabled = !complete();
    if (complete()) submitBtn.focus();
  }
  function choose(val) {
    const c = crits[active];
    if (!c) return;
    const inp = c.querySelector('input[value="' + val + '"]');
    if (inp) inp.checked = true;
    const next = crits.findIndex((x, k) => k > active && !x.querySelector('input:checked'));
    if (next >= 0) setActive(next);
    else { const first = crits.findIndex(x => !x.querySelector('input:checked')); if (first >= 0) setActive(first); }
    refresh();
  }
  crits.forEach((c, k) => c.addEventListener('change', () => { setActive(k); choose(c.querySelector('input:checked').value); }));
  crits.forEach((c, k) => c.addEventListener('click', () => setActive(k)));

  function playAll() { wantPlay = true; vids.forEach(v => { if (!ended(v)) v.play().catch(() => {}); }); playBtn.textContent = 'Pause'; }
  function pauseAll() { wantPlay = false; vids.forEach(v => v.pause()); playBtn.textContent = 'Play'; }
  function restart() { vids.forEach(v => { v.currentTime = 0; }); if (wantPlay) playAll(); }
  function seekTo(t) { vids.forEach(v => { v.currentTime = Math.min(t, (v.duration || t) - 0.001); }); }
  function step(d) { pauseAll(); seekTo(Math.max(0, Math.min(dur(), vL.currentTime + d))); }
  function setAudio(which) {
    audio = which;
    vL.muted = which !== 'left'; vR.muted = which !== 'right';
    document.getElementById('boxL').classList.toggle('aud', which === 'left');
    document.getElementById('boxR').classList.toggle('aud', which === 'right');
    document.querySelectorAll('[data-audio]').forEach(b => b.classList.toggle('primary', b.dataset.audio === which));
  }

  vids.forEach(v => {
    v.addEventListener('ended', () => {
      if (vids.every(ended)) { if (loopBox.checked && wantPlay) restart(); else { wantPlay = false; playBtn.textContent = 'Play'; } }
    });
    v.addEventListener('error', () => { const e = v.parentElement.querySelector('.err'); if (e) e.style.display = 'block'; });
  });
  // start together once both can play
  let ready = 0;
  vids.forEach(v => v.addEventListener('canplay', () => { ready++; if (ready === 2) playAll(); }, { once: true }));
  // keep the two clocks together and update the scrubber
  setInterval(() => {
    const d = dur();
    if (d) { seek.max = d; if (!seek.matches(':active')) seek.value = Math.max(vL.currentTime, vR.currentTime); }
    timeLbl.textContent = Math.max(vL.currentTime, vR.currentTime).toFixed(2) + ' / ' + (d || 0).toFixed(2) + ' s';
    if (wantPlay && !vL.paused && !vR.paused && !ended(vL) && !ended(vR) && Math.abs(vL.currentTime - vR.currentTime) > 0.07) {
      vR.currentTime = vL.currentTime;
    }
  }, 200);
  seek.addEventListener('input', () => seekTo(parseFloat(seek.value)));
  playBtn.addEventListener('click', () => (wantPlay && !vids.every(v => v.paused)) ? pauseAll() : (vids.every(ended) ? (wantPlay = true, restart()) : playAll()));
  document.getElementById('restart').addEventListener('click', () => { wantPlay = true; restart(); });
  document.querySelectorAll('[data-audio]').forEach(b => b.addEventListener('click', () => setAudio(b.dataset.audio)));

  form.addEventListener('submit', e => {
    if (!complete()) { e.preventDefault(); return; }
    document.getElementById('duration_ms').value = Math.round(performance.now() - t0);
  });

  document.addEventListener('keydown', e => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'textarea' || (tag === 'input' && e.target.type === 'text')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === '1') choose('left');
    else if (k === '2') choose('right');
    else if (k === '=' || k === '0' || k === '3') choose('tie');
    else if (k === ' ') playBtn.click();
    else if (k === 'r' || k === 'R') { wantPlay = true; restart(); }
    else if (k === 'l' || k === 'L') loopBox.checked = !loopBox.checked;
    else if (k === ',') step(-1 / 24);
    else if (k === '.') step(1 / 24);
    else if (k === 'a' || k === 'A') setAudio('left');
    else if (k === 's' || k === 'S') setAudio('right');
    else if (k === 'd' || k === 'D') setAudio('off');
    else if (k === 'ArrowUp') setActive(active - 1);
    else if (k === 'ArrowDown') setActive(active + 1);
    else if (k === 'u' || k === 'U') { const u = document.getElementById('undoform'); if (u) u.submit(); }
    else if (k === 'Enter') { if (complete()) form.requestSubmit(); }
    else return;
    e.preventDefault();
  });

  setActive(0); setAudio('off'); refresh();
})();
