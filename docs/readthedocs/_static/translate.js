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
    return toggle;
  }

  /** 把切换按钮插到每一处「上一页」后面；页面没有该按钮时退回侧栏。
   *
   * 主题在**正文顶部与页脚各渲染一处**「上一页 / 下一页」（配置里的
   * prev_next_buttons_location = both），只插一处会让另一半页面看起来没有按钮。 */
  function buildButton(english) {
    var prevLinks = Array.prototype.slice.call(document.querySelectorAll('a[rel="prev"]'));
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

  document.addEventListener('DOMContentLoaded', function () {
    var english = isEnglish();
    if (english) {
      // 已切换过：本次加载就要让组件把页面翻成英文，插图也换成英文图集。
      protectVerbatim();
      swapFigures();
      loadWidget();
    }
    buildButton(english);
  });
})();
