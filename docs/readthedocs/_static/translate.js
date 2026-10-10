/* 中英切换：走 Google 网站翻译（googtrans cookie + 按需加载 element.js）。
 *
 * 为什么是「cookie + 按需加载」而不是直接嵌官方组件：
 *   1. 官方组件会在每个页面加载第三方脚本，而我们只希望切换过英文的读者加载它；
 *   2. googtrans cookie 是官方组件的「记住选择」机制，切换后重新加载页面时组件会
 *      自动翻译，所以按钮只需要写 cookie + reload。
 *
 * 加载时机：英文模式下本脚本一开始执行（head 解析阶段）就去拉 element.js，不等
 * DOMContentLoaded——head 里的 MathJax 是 defer 的，DOMContentLoaded 要等它下载并执行完，
 * 线上实测把这一步拖到 3.3–12.8 s。提前注入后 element.js 请求落在 0.6–2.1 s。
 *
 * 注意：Google 翻译是让 Google 的服务器来抓取页面，因此只有站点能被公网访问时才
 * 有效（RTD 部署后可用）。本地预览时按钮置灰并给出提示，本地想看英文可以用浏览器
 * 自带的翻译。
 */
(function () {
  'use strict';

  var COOKIE = 'googtrans';
  var SOURCE = 'zh-CN';
  var TARGET = 'en';
  var WIDGET_ID = 'google-translate-script';

  // 机器翻译不该动的节点：写进代码块与公式里的中文要原样留着。
  var VERBATIM_SELECTOR = 'pre, code, .highlight, .math, mjx-container';

  // 代码块注释词典（覆盖全站全部含中文的代码块文本节点，与英文图集同一条人工翻译
  // 产线）。代码块被 notranslate 保护、机器翻译碰不到，所以英文版靠这里按词典替换。
  var CODE_DICT_URL = '_static/code-en.json';

  // 取样的汉字数掉到起始值的这个比例以下、且首个标题已翻成英文，就算「翻完」。不能要求
  // 「一个汉字都不剩」：受保护的代码块本来就留着中文，Google 也**只翻一部分正文**——ch01 上
  // 线上与本地一致地留下 417 个汉字（约 47% 的取样节点）永久不动，占正文一成左右。
  var RESIDUE_RATIO = 0.7;

  // 从发起 element.js 请求算起，等这么久还没翻完就认为 Google 卡住了（实测翻完中位 14.6 s、
  // 最长 22.0 s，另有 45 s 与 90 s 都没翻完的失败样本），此时给读者一条退回中文的出口。
  var SLOW_LIMIT_MS = 30000;

  // 兜底提示出现后继续慢速轮询这么久，之后停表（兜底链接留在页面上）。
  var SLOW_WATCH_MS = 120000;

  // 发起 element.js 请求的时刻，超时从这里算起。
  var requestedAt = 0;

  /** 读取某个 cookie。 */
  function readCookie(name) {
    var match = document.cookie.match(new RegExp('(?:^|;\\s*)' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[1]) : '';
  }

  /** 判断当前是否处于英文模式。 */
  function isEnglish() {
    return readCookie(COOKIE).indexOf('/' + SOURCE + '/' + TARGET) !== -1;
  }

  /** 当前页面能否被 Google 抓取（本地/内网地址不行）。 */
  function isReachable() {
    if (location.protocol === 'file:') {
      return false;
    }
    var host = location.hostname;
    return !(host === 'localhost' || host === '127.0.0.1' || host === '' ||
             /\.local$/.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) ||
             /^172\.(1[6-9]|2\d|3[01])\./.test(host));
  }

  /** 写入 cookie（同时写一份带域的，覆盖子域场景）。 */
  function writeCookie(value) {
    var base = COOKIE + '=' + value + ';path=/';
    document.cookie = base;
    if (location.hostname) {
      document.cookie = base + ';domain=' + location.hostname;
    }
  }

  /** 给代码块与公式打 notranslate：机器翻译不该动这些内容。
   *
   * 右栏的「本页目录」不在保护范围内——它是页面正文，英文读者需要它也被译成英文
   * （Google 只改写文本节点，``href`` 里的锚点原样保留，跳转照常可用）。
   */
  function protectVerbatim() {
    Array.prototype.forEach.call(document.querySelectorAll(VERBATIM_SELECTOR),
                                 function (node) {
      node.classList.add('notranslate');
    });
  }

  /** 把插图换成英文图集里的同名文件（构建时按双语词典重新编译的那套）。
   *
   * 图内文字是矢量轮廓，机器翻译碰不到（Google 只改写 HTML 文本节点），所以英文版
   * 只能靠构建期重新编译一套 SVG。这里只换存在的那几张：加载失败就回退原图。 */
  function swapFigures() {
    var EN_DIR = '_static/figures-en/';
    Array.prototype.forEach.call(document.querySelectorAll('.rst-content img'), function (image) {
      var source = image.getAttribute('src') || '';
      var match = source.match(/(?:^|\/)_images\/([^/?#]+)$/);
      if (!match) {
        return;
      }
      var original = source;
      image.addEventListener('error', function () {
        // 没有英文版的图（未全部命中词典）继续用中文原图。
        if (image.getAttribute('src') !== original) {
          image.setAttribute('src', original);
        }
      });
      image.setAttribute('src', EN_DIR + match[1]);
    });
  }

  /** 对容器内文本节点按词典整段替换（trim 后命中，保留前后空白）。
   *
   * 先收集再替换；replacement 用函数形式——译文里的 ``$&``、``$'`` 之类在字符串
   * 形式的 replacement 里会被当成替换模式解释，函数形式则按字面返回。
   */
  function applyDict(selector, dict) {
    var cjk = /[\u4e00-\u9fff]/;
    Array.prototype.forEach.call(document.querySelectorAll(selector), function (root) {
      // 先收集再替换，避免边遍历边改。
      var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
      var nodes = [];
      while (walker.nextNode()) {
        nodes.push(walker.currentNode);
      }
      nodes.forEach(function (node) {
        var text = node.nodeValue;
        if (!cjk.test(text)) {
          return;
        }
        var trimmed = text.trim();
        if (Object.prototype.hasOwnProperty.call(dict, trimmed)) {
          node.nodeValue = text.replace(trimmed, function () {
            return dict[trimmed];
          });
        }
      });
    });
  }

  /** 按词典把代码块里的中文注释换成英文（英文图集的同构机制）。
   *
   * 代码块在 ``notranslate`` 保护下机器翻译碰不到；图内文字走「构建期重编译一套
   * SVG」，代码块则由这里在英文模式下拉取 ``_static/code-en.json``、对 ``pre`` 里的
   * 文本节点按 key 替换。key 是文本节点 trim 后的原文，命中才换并保留前后空白（缩进
   * 不动）；未命中的注释保持中文——读者看到的是完整注释，而不是半中半英。本地预览
   * 同样生效，不依赖 Google 可达。代码块不受 MathJax 影响，走 fetch 即可。 */
  function swapCodeComments() {
    if (!window.fetch) {
      return;
    }
    fetch(CODE_DICT_URL).then(function (response) {
      if (!response.ok) {
        return null;
      }
      return response.json();
    }).then(function (dict) {
      if (!dict) {
        return;
      }
      applyDict('pre', dict);
    }).catch(function () {
      // 词典拉不下来就保持中文：英文体验可以降级，代码块不能坏。
    });
  }

  /** 按词典把公式 tex 源里的中文换成英文（``\text{}`` 注释之类）。
   *
   * 词典来自 ``math-en.js``（``conf.py`` 里排在本脚本之前同步加载），所以这里是
   * 同步替换：MathJax 是 defer 的，在 ``interactive`` 之后就开始渲染、把 tex 源文本
   * 节点换成 ``mjx-container``，fetch 异步词典可能输给它——公式替换必须抢在渲染前。
   * key 是整段公式 tex 源（trim 后），由 ``tools/code_dict.py`` 的骨架校验保证译文
   * 只动了中文、LaTeX 结构逐字未变。 */
  function swapMathText() {
    if (window.LEETCUDA_MATH_EN) {
      applyDict('.math', window.LEETCUDA_MATH_EN);
    }
  }

  /** 按需加载官方翻译组件（用到才加载，避免每页引入第三方脚本）。
   *
   * 英文模式下在 head 解析阶段就被调用，此时 body 还不存在，容器与脚本挂到 head 上。
   */
  function loadWidget() {
    if (document.getElementById(WIDGET_ID)) {
      return;
    }
    requestedAt = Date.now();
    var host = document.head || document.body || document.documentElement;
    var holder = document.createElement('div');
    holder.id = 'google_translate_element';
    holder.style.display = 'none';
    host.appendChild(holder);

    window.googleTranslateElementInit = function () {
      /* global google */
      // 组件就绪时 DOM 可能只解析了一半，先把此刻已有的代码块与公式保护上
      // （DOMContentLoaded 那一遍会补全剩下的）；机器翻译不该动这些内容。
      protectVerbatim();
      new google.translate.TranslateElement({
        pageLanguage: SOURCE,
        includedLanguages: TARGET + ',' + SOURCE,
        autoDisplay: false
      }, 'google_translate_element');
    };

    var script = document.createElement('script');
    script.id = WIDGET_ID;
    script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    host.appendChild(script);
  }

  /** 造一个切换按钮（与主题的「上一页 / 下一页」同款：`btn btn-neutral` + 图标）。 */
  function createToggle(english) {
    var toggle = document.createElement('a');
    toggle.className = 'btn btn-neutral rtd-translate-toggle';
    toggle.setAttribute('role', 'button');
    toggle.setAttribute('tabindex', '0');

    var icon = document.createElement('i');
    icon.className = 'fa fa-language';
    icon.setAttribute('aria-hidden', 'true');
    toggle.appendChild(icon);
    toggle.appendChild(document.createTextNode(english ? ' 中文' : ' English'));

    function activate() {
      if (!isReachable()) {
        // 外观与主题按钮完全一致（不做置灰，否则与旁边的「上一页」风格不一），
        // 本地预览点了给一条提示，说明为什么这里用不了。
        showHint(toggle);
        return;
      }
      if (english) {
        writeCookie('/' + SOURCE + '/' + SOURCE);
      } else {
        protectVerbatim();
        writeCookie('/' + SOURCE + '/' + TARGET);
      }
      location.reload();
    }

    toggle.title = isReachable()
      ? (english ? '切回中文' : '用 Google 翻译阅读英文版')
      : '本地预览无法使用 Google 翻译（需要站点可被公网访问）；' +
        '本地可先用浏览器自带的翻译功能';
    toggle.addEventListener('click', function (event) {
      event.preventDefault();
      activate();
    });
    toggle.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate();
      }
    });
    return toggle;
  }

  /** 在按钮旁显示一条临时提示（本地预览下用不了 Google 翻译时）。 */
  function showHint(anchor) {
    if (anchor.nextElementSibling &&
        anchor.nextElementSibling.classList.contains('rtd-translate-hint')) {
      return;
    }
    var hint = document.createElement('span');
    hint.className = 'rtd-translate-hint';
    hint.textContent = 'Google 翻译需要站点可被公网访问，部署到 RTD 后即可用';
    anchor.insertAdjacentElement('afterend', hint);
    window.setTimeout(function () {
      hint.remove();
    }, 8000);
  }

  /** 取样正文里还剩多少汉字（跨整页等距取样，剔除受保护节点里的中文）。
   *
   * 不敢只取开头那几十个：长章节的取样节点能到 500 个，只看开头会在页面下半仍是中文时报
   * 「已切换」。
   */
  function remainingCjk() {
    var nodes = document.querySelectorAll(
      '.rst-content h1, .rst-content h2, .rst-content h3, .rst-content p');
    var stride = Math.max(1, Math.ceil(nodes.length / 30));
    var left = 0;
    for (var index = 0; index < nodes.length; index += stride) {
      var node = nodes[index];
      if (node.closest(VERBATIM_SELECTOR)) {
        continue;
      }
      var copy = node.cloneNode(true);
      Array.prototype.forEach.call(copy.querySelectorAll(VERBATIM_SELECTOR),
                                   function (child) {
        child.remove();
      });
      left += (copy.textContent.match(/[\u4e00-\u9fff]/g) || []).length;
    }
    return left;
  }

  /** 第一个标题是否已经翻成英文（正文替换只有一两批，它落下来说明替换已经发生）。 */
  function headingTranslated() {
    var heading = document.querySelector(
      '.rst-content h1, .rst-content h2, .rst-content h3');
    return !heading || !/[\u4e00-\u9fff]/.test(heading.textContent);
  }

  /** 造一条翻译状态提示，挂在切换按钮后面。文案用英文：只在英文模式下出现，而且 Google
   *  未必会把后加进来的节点再翻一遍。 */
  function createStatus(anchor) {
    var status = document.createElement('span');
    status.className = 'rtd-translate-status';
    status.setAttribute('role', 'status');
    status.textContent = 'Loading the English version…';
    anchor.insertAdjacentElement('afterend', status);
    return status;
  }

  /** 盯正文什么时候变成英文，顺便维护等待中的提示。
   *
   * 站点侧看不到 Google 的进度：它插的横幅 iframe 比正文替换早十几秒，唯一可靠的判据是正文
   * 自己变没变。翻完中位 14.6 s（9.0–22.0 s），也出现过一直不翻的卡死，所以超过
   * ``SLOW_LIMIT_MS`` 就给读者一条退回中文的出口，别让英文读者对着中文干等。
   */
  function watchTranslation(anchor) {
    var status = createStatus(anchor);
    var deadline = (requestedAt || Date.now()) + SLOW_LIMIT_MS;
    var stopAt = deadline + SLOW_WATCH_MS;
    var initial = Math.max(remainingCjk(), 1);
    var first = true;
    var slow = false;
    function tick() {
      // 首个标题翻成英文说明替换已经发生；光看它不够——正文分批落地，它可能先到，所以要求
      // 取样汉字也掉下来。第一次取样时页面还没被翻（此时标题仍是中文），所以 `first` 只在
      // 「进来时就已经翻好」这种情况下直接判完成。
      if (headingTranslated() &&
          (first || remainingCjk() <= initial * RESIDUE_RATIO)) {
        status.textContent = 'Switched to the English version';
        window.setTimeout(function () {
          status.remove();
        }, 2500);
        return;
      }
      first = false;
      if (!slow && Date.now() > deadline) {
        // 兜底之后继续慢速轮询：Google 只是慢、后来又翻完了的话，提示要跟着收掉。
        slow = true;
        status.textContent = 'Google Translate is slow, ';
        var fallback = document.createElement('a');
        fallback.href = '#';
        fallback.textContent = 'read in Chinese';
        fallback.addEventListener('click', function (event) {
          event.preventDefault();
          writeCookie('/' + SOURCE + '/' + SOURCE);
          location.reload();
        });
        status.appendChild(fallback);
      }
      if (slow && Date.now() > stopAt) {
        return;
      }
      window.setTimeout(tick, slow ? 2000 : 400);
    }
    tick();
  }

  /** 收集页面上所有的「上一页」按钮。
   *
   * 主题按 ``prev_next_buttons_location = both`` 渲染两处：正文顶部的
   * ``.rst-breadcrumbs-buttons`` 与页脚的 ``.rst-footer-buttons``。**顶部那处的锚点没有
   * ``rel="prev"``**（只有页脚有），只按 ``rel="prev"`` 找会漏掉用户实际看到的那一处。 */
  function prevButtons() {
    var selectors = ['.rst-breadcrumbs-buttons a.float-left',
                     '.rst-footer-buttons a[rel="prev"]',
                     'a[rel="prev"]'];
    var found = [];
    selectors.forEach(function (selector) {
      Array.prototype.forEach.call(document.querySelectorAll(selector), function (node) {
        if (found.indexOf(node) === -1) {
          found.push(node);
        }
      });
    });
    return found;
  }

  /** 把切换按钮放进页面的按钮行；连按钮行都没有的页面才退回侧栏。
   *
   * 三种落位：
   * - **有「上一页」的页面**：插在每一处「上一页」后面。主题把上一页/下一页渲染在顶部与
   *   页脚两处（`conf.py` 的 `prev_next_buttons_location = both`），所以这里也是两处。
   * - **没有「上一页」的页面**（首页）：插进那两处按钮行的右侧按钮组、落在「下载 PDF」
   *   右边，左侧导航栏里不再放。
   * - **连按钮行都没有的页面**（`genindex` / `search`）：退回侧栏——模板给出的
   *   `.rtd-sidebar-actions`（「下载 PDF」也在那一行），没有该容器时才自建
   *   `.rtd-translate` 包装（旧版页面产物）。
   *
   * ⚠️ 按钮行是浮动布局：`float: right` 的元素**先出现的贴右缘**，后出现的挤到它左边。
   * 所以「显示在下载按钮右边」在 DOM 上得插在它**前面**，并带上 `rtd-translate-toggle-right`
   * （给它在 `_static/custom.css` 里加了 `float: right`）。
   */
  function buildButton(english) {
    var prevLinks = prevButtons();
    prevLinks.forEach(function (prev) {
      if (prev.parentElement &&
          prev.parentElement.querySelector('.rtd-translate-toggle')) {
        return;
      }
      prev.insertAdjacentElement('afterend', createToggle(english));
    });
    if (prevLinks.length) {
      return;
    }
    var placed = false;
    var rows = document.querySelectorAll('.rst-breadcrumbs-buttons, .rst-footer-buttons');
    Array.prototype.forEach.call(rows, function (row) {
      var pdf = row.querySelector('.rtd-pdf-download');
      if (!pdf || row.querySelector('.rtd-translate-toggle')) {
        return;
      }
      var toggle = createToggle(english);
      toggle.classList.add('rtd-translate-toggle-right');
      pdf.insertAdjacentElement('beforebegin', toggle);
      placed = true;
    });
    if (placed) {
      return;
    }
    var actions = document.querySelector('.rtd-sidebar-actions');
    if (actions) {
      if (!actions.querySelector('.rtd-translate-toggle')) {
        actions.appendChild(createToggle(english));
      }
      return;
    }
    var sidebar = document.querySelector('.wy-side-nav-search') ||
                  document.querySelector('.wy-nav-side');
    if (!sidebar) {
      return;
    }
    var wrapper = document.createElement('div');
    wrapper.className = 'rtd-translate';
    wrapper.appendChild(createToggle(english));
    sidebar.appendChild(wrapper);
  }

  /** DOM 一解析完就跑。
   *
   * ``interactive`` 早于 ``DOMContentLoaded``：head 里的 MathJax 是 defer 的，DOMContentLoaded
   * 要等它下载并执行完（线上实测 3.3–12.8 s），而这些改动只需要 DOM 结构，不必等它。
   */
  function whenDomReady(callback) {
    if (document.readyState !== 'loading') {
      callback();
      return;
    }
    var fired = false;
    document.addEventListener('readystatechange', function () {
      if (!fired && document.readyState !== 'loading') {
        fired = true;
        callback();
      }
    });
  }

  // 已切换过英文：head 解析阶段就发起组件请求，别等 DOMContentLoaded（见文件头）。
  if (isEnglish()) {
    loadWidget();
  }

  whenDomReady(function () {
    var english = isEnglish();
    if (english) {
      // 内容侧的准备：保护代码块与公式、把插图换成英文图集、代码块注释与公式 tex
      // 源换成英文词典（公式必须同步抢在 MathJax 渲染前，见 swapMathText）。
      protectVerbatim();
      swapFigures();
      swapCodeComments();
      swapMathText();
      // MathJax 是在 `interactive` 之后、`DOMContentLoaded` 之前才把 `mjx-container` 建出来
      // 的（实测受保护节点 131 → 244），那时再补一遍，别让机器翻译动公式。
      document.addEventListener('DOMContentLoaded', protectVerbatim);
    }
    buildButton(english);
    if (english) {
      var toggle = document.querySelector('.rtd-translate-toggle');
      if (toggle) {
        watchTranslation(toggle);
      }
    }
  });
})();
