# FIG-25-1 CuTe 块结构对照（drawio 新建，2026-09-14）

- 内容：flash_attn.cuh CuTe 块四层结构——共享层（FA2/FA3 traits 并排 + 五工具函数
  横条）-> 三实现卡并排（cp.async / TMA+WS / FA3 双 consumer，各标线程数与流水特征）
  -> 实现4 smoke 横贯卡（41 行最小闭环）；实线=复用同一份布局类型，虚线=公共底座自检。
- 执行模式：主 agent 直做（gen.py -> drawio-headless -s 3 -> PIL 特征色/色块程序化
  验收，view_image 显示层缓存损坏不可用），coordinator 自审。
- 迭代记录（3 处 XML 非法字符教训）：v1 traits 卡 title `FlashAttnNCuTeTraits<D>`
  与 v2 实现卡 `wait<N>` 的裸泛型尖括号破坏 well-formedness，drawio DOMParser 静默
  吞掉其后全部元素（渲染截断在断点、无任何报错）-> 全部转义 `&lt;D&gt;`/`&lt;N&gt;`
  后四层齐全（特征色 BLUE/GREEN/ORANGE/BROWN 全 FOUND，ratio 0.598）。
- 规格：画布 1150x~640；文字卡字号 >= 12；配色 蓝布局/橙绿调度。
