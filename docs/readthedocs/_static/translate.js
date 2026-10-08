/* 中英切换：走 Google 网站翻译（googtrans cookie + 按需加载 element.js）。
 *
 * 为什么是「cookie + 按需加载」而不是直接嵌官方组件：
 *   1. 官方组件会在每个页面加载第三方脚本，而我们只希望切换过英文的读者加载它；
 *   2. googtrans cookie 是官方组件的「记住选择」机制，切换后重新加载页面时组件会
 *      自动翻译，所以按钮只需要写 cookie + reload。
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

  /** 给代码块与公式打 notranslate：机器翻译不该动这些内容。 */
  function protectVerbatim() {
    var selectors = ['pre', 'code', '.highlight', '.math', 'mjx-container', '.rtd-localtoc a'];
    selectors.forEach(function (selector) {
      Array.prototype.forEach.call(document.querySelectorAll(selector), function (node) {
        node.classList.add('notranslate');
      });
    });
  }

  /** 按需加载官方翻译组件（用到才加载，避免每页引入第三方脚本）。 */
  function loadWidget() {
    if (document.getElementById(WIDGET_ID)) {
      return;
    }
    var holder = document.createElement('div');
    holder.id = 'google_translate_element';
    holder.style.display = 'none';
    document.body.appendChild(holder);

    window.googleTranslateElementInit = function () {
      /* global google */
      new google.translate.TranslateElement({
        pageLanguage: SOURCE,
        includedLanguages: TARGET + ',' + SOURCE,
        autoDisplay: false
      }, 'google_translate_element');
    };

    var script = document.createElement('script');
    script.id = WIDGET_ID;
    script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    document.body.appendChild(script);
  }

  /** 生成切换按钮：放在页脚导航行里「上一页」旁边，没有该行时退回侧栏。 */
  function buildButton(english) {
    var prev = document.querySelector('.rst-footer-buttons a[rel="prev"]');
    var sidebar = document.querySelector('.wy-side-nav-search') ||
                  document.querySelector('.wy-nav-side');
    if (!prev && !sidebar) {
      return;
    }

    // 与主题的「上一页 / 下一页」同款：`btn btn-neutral` + FontAwesome 图标
    // （主题用的是 <a class="btn btn-neutral"><span class="fa fa-…"></span> 文本</a>）。
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
      if (english) {
        writeCookie('/' + SOURCE + '/' + SOURCE);
      } else {
        protectVerbatim();
        writeCookie('/' + SOURCE + '/' + TARGET);
      }
      location.reload();
    }

    if (!isReachable()) {
      toggle.classList.add('rtd-translate-disabled');
      toggle.setAttribute('aria-disabled', 'true');
      toggle.removeAttribute('tabindex');
      toggle.title = '本地预览无法使用 Google 翻译（需要站点可被公网访问）；' +
                     '本地可先用浏览器自带的翻译功能';
    } else {
      toggle.title = english ? '切回中文' : '用 Google 翻译阅读英文版';
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
    }

    if (prev) {
      prev.insertAdjacentElement('afterend', toggle);
      return;
    }
    var wrapper = document.createElement('div');
    wrapper.className = 'rtd-translate';
    wrapper.appendChild(toggle);
    sidebar.appendChild(wrapper);
  }

  document.addEventListener('DOMContentLoaded', function () {
    var english = isEnglish();
    if (english) {
      // 已切换过：本次加载就要让组件把页面翻成英文。
      protectVerbatim();
      loadWidget();
    }
    buildButton(english);
  });
})();
