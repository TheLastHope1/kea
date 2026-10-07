// Kea report charts: dependency-free SVG line and stacked-area charts.
// Each <figure class="chart"> carries its data as JSON; this script draws it at
// the figure's real width, redraws on resize, and adds a crosshair readout that
// follows the pointer or the arrow keys. Every value is also in the table view.
(function () {
  "use strict";
  var SVG = "http://www.w3.org/2000/svg";
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function el(name, attrs, parent) {
    var node = document.createElementNS(SVG, name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function compact(v, prefix) {
    var a = Math.abs(v);
    var s = a >= 1e6 ? (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M"
      : a >= 1e3 ? (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k"
      : v.toFixed(a >= 100 ? 0 : 1);
    return (prefix || "") + s.replace(".0M", "M").replace(".0k", "k");
  }

  var FORMATS = {
    money: { tick: function (v) { return compact(v, "$"); },
             value: function (v) { return "$" + Math.round(v).toLocaleString("en-US"); } },
    index: { tick: function (v) { return v.toFixed(0); },
             value: function (v) { return v.toFixed(1); } },
    pct: { tick: function (v) { return Math.round(v * 100) + "%"; },
           value: function (v) { return (v * 100).toFixed(1) + "%"; } }
  };

  function niceLinear(min, max, target) {
    var span = max - min || Math.abs(max) || 1;
    var raw = span / target;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; })
      .find(function (s) { return span / s <= target; });
    var ticks = [];
    for (var t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) ticks.push(+t.toFixed(10));
    return ticks;
  }

  function niceLog(min, max) {
    var sets = [[1], [1, 2, 5], [1, 1.5, 2, 3, 5, 7]];
    var best = [];
    for (var i = 0; i < sets.length; i++) {
      var ticks = [];
      for (var e = Math.floor(Math.log10(min)) - 1; e <= Math.ceil(Math.log10(max)); e++) {
        sets[i].forEach(function (m) {
          var v = m * Math.pow(10, e);
          if (v >= min && v <= max) ticks.push(v);
        });
      }
      best = ticks;
      if (ticks.length >= 4) break;
    }
    return best;
  }

  function dateTicks(dates, width) {
    var first = dates[0], last = dates[dates.length - 1];
    var years = (last - first) / 3.15576e10;
    var maxTicks = Math.max(2, Math.floor(width / 90));
    var ticks = [];
    if (years >= 2) {
      var stepYears = [1, 2, 5, 10, 20].find(function (s) { return years / s <= maxTicks; }) || 25;
      for (var y = first.getUTCFullYear() + 1; y <= last.getUTCFullYear(); y++) {
        if (y % stepYears === 0) ticks.push({ t: Date.UTC(y, 0, 1), label: String(y) });
      }
    } else {
      var months = years * 12;
      var stepMonths = [1, 2, 3, 6].find(function (s) { return months / s <= maxTicks; }) || 12;
      var d = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 1));
      for (; d <= last; d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1))) {
        if (d.getUTCMonth() % stepMonths === 0) {
          var label = MONTHS[d.getUTCMonth()] + (d.getUTCMonth() === 0 || ticks.length === 0 ? " " + d.getUTCFullYear() : "");
          ticks.push({ t: d.getTime(), label: label });
        }
      }
    }
    return ticks;
  }

  function longDate(d) {
    return d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear();
  }

  function draw(figure) {
    var spec = JSON.parse(figure.querySelector("script[type='application/json']").textContent);
    var host = figure.querySelector(".chart-plot");
    var width = Math.max(host.clientWidth, 280);
    var height = spec.height || 300;
    var fmt = FORMATS[spec.format] || FORMATS.index;
    var stacked = spec.kind === "stack";
    var dates = spec.dates.map(function (s) { return new Date(s + "T00:00:00Z"); });
    var series = spec.series;
    var labelled = !stacked && series.length <= 4;
    var narrow = width < 560; // phones: end labels show values only; the legend names them
    var m = { top: 12, right: !labelled ? 16 : narrow ? 64 : Math.min(150, width * 0.3), bottom: 30, left: 52 };
    var w = width - m.left - m.right, h = height - m.top - m.bottom;

    // Stacked areas plot cumulative shares; lines plot the values themselves.
    var layers = series.map(function (s) { return s.values.slice(); });
    if (stacked) {
      for (var i = 1; i < layers.length; i++) {
        layers[i] = layers[i].map(function (v, j) { return v + layers[i - 1][j]; });
      }
    }
    var finite = [].concat.apply([], layers).filter(function (v) { return v !== null && isFinite(v); });
    var lo = stacked ? 0 : Math.min.apply(null, finite), hi = stacked ? 1 : Math.max.apply(null, finite);
    if (spec.zero === "top") hi = 0;
    if (!spec.log && !stacked && spec.zero !== "top") { var pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad; }
    if (spec.log) { lo = lo * 0.94; hi = hi * 1.06; }
    if (spec.zero === "top") lo = lo * 1.08;

    var t0 = dates[0].getTime(), t1 = dates[dates.length - 1].getTime();
    var x = function (t) { return m.left + (t - t0) / (t1 - t0 || 1) * w; };
    var y = spec.log
      ? function (v) { return m.top + (Math.log(hi) - Math.log(v)) / (Math.log(hi) - Math.log(lo)) * h; }
      : function (v) { return m.top + (hi - v) / (hi - lo || 1) * h; };

    var previous = host.querySelector("svg");
    if (previous) previous.remove();
    var svg = el("svg", { width: width, height: height, viewBox: "0 0 " + width + " " + height,
      role: "img", "aria-label": spec.title || "chart", class: "chart-svg" });
    host.insertBefore(svg, host.firstChild);

    var yTicks = stacked ? [0, 0.25, 0.5, 0.75, 1] : spec.log ? niceLog(lo, hi) : niceLinear(lo, hi, Math.max(3, Math.round(h / 60)));
    var grid = el("g", { class: "grid" }, svg);
    yTicks.forEach(function (v) {
      var yy = Math.round(y(v)) + 0.5;
      el("line", { x1: m.left, x2: m.left + w, y1: yy, y2: yy }, grid);
      var label = el("text", { x: m.left - 8, y: yy, dy: "0.32em", "text-anchor": "end", class: "tick" }, svg);
      label.textContent = stacked ? Math.round(v * 100) + "%" : fmt.tick(v);
    });
    dateTicks(dates, w).forEach(function (tick) {
      if (tick.t < t0 || tick.t > t1) return;
      var xx = Math.round(x(tick.t)) + 0.5;
      el("line", { x1: xx, x2: xx, y1: m.top + h, y2: m.top + h + 4, class: "axis" }, svg);
      var label = el("text", { x: xx, y: m.top + h + 18, "text-anchor": "middle", class: "tick" }, svg);
      label.textContent = tick.label;
    });
    var base = spec.zero === "top" ? y(0) : m.top + h;
    el("line", { x1: m.left, x2: m.left + w, y1: Math.round(base) + 0.5, y2: Math.round(base) + 0.5, class: "axis" }, svg);

    function path(values) {
      var d = "", pen = false;
      values.forEach(function (v, j) {
        if (v === null || !isFinite(v)) { pen = false; return; }
        d += (pen ? "L" : "M") + x(dates[j].getTime()).toFixed(1) + " " + y(v).toFixed(1);
        pen = true;
      });
      return d;
    }

    if (stacked) {
      layers.forEach(function (upper, i) {
        var lower = i ? layers[i - 1] : upper.map(function () { return 0; });
        var top = path(upper);
        var bottom = lower.map(function (v, j) {
          return "L" + x(dates[j].getTime()).toFixed(1) + " " + y(v).toFixed(1);
        }).reverse().join("");
        var area = el("path", { d: top + bottom + "Z", class: "stack-area" }, svg);
        area.style.fill = "var(" + series[i].color + ")";
        el("path", { d: top, class: "stack-gap" }, svg);
      });
    } else {
      series.forEach(function (s, i) {
        if (s.area) {
          var fill = el("path", { d: path(layers[i]) + "L" + x(t1) + " " + base + "L" + x(t0) + " " + base + "Z", class: "area-wash" }, svg);
          fill.style.fill = "var(" + s.color + ")";
        }
        var line = el("path", { d: path(layers[i]), class: "line" }, svg);
        line.style.stroke = "var(" + s.color + ")";
      });
    }

    // Direct end labels (<= 4 series), nudged apart with leader lines if they collide.
    if (labelled) {
      var ends = series.map(function (s, i) {
        var j = s.values.length - 1;
        while (j > 0 && (s.values[j] === null || !isFinite(s.values[j]))) j--;
        return { i: i, y: y(s.values[j]), yy: y(s.values[j]), v: s.values[j] };
      }).sort(function (a, b) { return a.y - b.y; });
      var gap = narrow ? 16 : 30;
      for (var k = 1; k < ends.length; k++) ends[k].yy = Math.max(ends[k].yy, ends[k - 1].yy + gap);
      var overflow = ends.length ? ends[ends.length - 1].yy - (m.top + h) : 0;
      if (overflow > 0) ends.forEach(function (e) { e.yy -= overflow; });
      ends.forEach(function (e) {
        var s = series[e.i], xe = x(t1);
        if (Math.abs(e.yy - e.y) > 2) el("path", { d: "M" + (xe + 6) + " " + e.y + "L" + (xe + 12) + " " + e.yy, class: "leader" }, svg);
        var dot = el("circle", { cx: xe, cy: e.y, r: 4, class: "end-dot" }, svg);
        dot.style.fill = "var(" + s.color + ")";
        var value = el("text", { x: xe + 14, y: narrow ? e.yy + 4 : e.yy - 3, class: "end-value" }, svg);
        value.textContent = narrow ? fmt.tick(e.v) : fmt.value(e.v);
        if (!narrow) {
          var name = el("text", { x: xe + 14, y: e.yy + 11, class: "end-name" }, svg);
          name.textContent = s.name;
        }
      });
    }

    // Crosshair readout: follows the pointer, or the arrow keys when focused.
    var cross = el("line", { y1: m.top, y2: m.top + h, class: "crosshair", visibility: "hidden" }, svg);
    var marks = stacked ? [] : series.map(function (s) {
      var c = el("circle", { r: 4, class: "end-dot", visibility: "hidden" }, svg);
      c.style.fill = "var(" + s.color + ")";
      return c;
    });
    var tip = figure.querySelector(".chart-tip");
    var index = -1;

    function show(j) {
      index = Math.max(0, Math.min(dates.length - 1, j));
      var xx = x(dates[index].getTime());
      cross.setAttribute("x1", xx); cross.setAttribute("x2", xx);
      cross.setAttribute("visibility", "visible");
      tip.textContent = "";
      var head = document.createElement("div");
      head.className = "tip-date";
      head.textContent = longDate(dates[index]);
      tip.appendChild(head);
      series.forEach(function (s, i) {
        var v = s.values[index];
        if (!stacked) {
          if (v === null || !isFinite(v)) { marks[i].setAttribute("visibility", "hidden"); return; }
          marks[i].setAttribute("cx", xx); marks[i].setAttribute("cy", y(v));
          marks[i].setAttribute("visibility", "visible");
        }
        var row = document.createElement("div");
        row.className = "tip-row";
        var key = document.createElement("span");
        key.className = stacked ? "key key-box" : "key";
        key.style.background = "var(" + s.color + ")";
        var strong = document.createElement("strong");
        strong.textContent = v === null || !isFinite(v) ? "n/a" : stacked ? (v * 100).toFixed(0) + "%" : fmt.value(v);
        var name = document.createElement("span");
        name.textContent = s.name;
        row.appendChild(key); row.appendChild(strong); row.appendChild(name);
        tip.appendChild(row);
      });
      tip.hidden = false;
      var left = xx + 14, room = host.clientWidth - tip.offsetWidth - 4;
      tip.style.left = (left > room ? Math.max(4, xx - tip.offsetWidth - 14) : left) + "px";
      tip.style.top = m.top + "px";
    }

    function hide() {
      index = -1;
      cross.setAttribute("visibility", "hidden");
      marks.forEach(function (c) { c.setAttribute("visibility", "hidden"); });
      tip.hidden = true;
    }

    function nearest(clientX) {
      var t = t0 + (clientX - svg.getBoundingClientRect().left - m.left) / w * (t1 - t0);
      var a = 0, b = dates.length - 1;
      while (b - a > 1) { var mid = (a + b) >> 1; if (dates[mid].getTime() < t) a = mid; else b = mid; }
      return t - dates[a].getTime() < dates[b].getTime() - t ? a : b;
    }

    svg.addEventListener("pointermove", function (e) { show(nearest(e.clientX)); });
    svg.addEventListener("pointerleave", hide);
    figure.onkeydown = function (e) {
      var step = e.shiftKey ? 20 : 1;
      if (e.key === "ArrowRight") { show(index < 0 ? dates.length - 1 : index + step); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { show(index < 0 ? dates.length - 1 : index - step); e.preventDefault(); }
      else if (e.key === "Escape") hide();
    };
    figure.onblur = hide;
  }

  function init() {
    document.querySelectorAll("figure.chart").forEach(function (figure) {
      try { draw(figure); } catch (err) { figure.classList.add("chart-failed"); return; }
      var width = figure.clientWidth;
      if (window.ResizeObserver) {
        new ResizeObserver(function () {
          if (Math.abs(figure.clientWidth - width) > 4) { width = figure.clientWidth; draw(figure); }
        }).observe(figure);
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
