# 【待办】外部咨询与 WCL 投稿前审阅计划

> 状态：**已激活——启动条件部分满足，纳入 wcl/ 管理**  
> 创建日期：2026-07-30  
> 目标：在不依赖本校老师署名的情况下，获得无线通信、强化学习、统计分析和 IEEE
> 写作方面的专业意见，提高稿件技术严谨性与 WCL 投稿质量。

## 1. 启动条件

以下条件已于 2026-08-03 满足：K 消融完成、正式报告与核心图表生成、论文创新叙事
冻结（"模型驱动规划与真实交互样本效率"）。仍待满足：

- [ ] warm-up 与 Case 2 的执行范围确定；
- [ ] 准备可供外部审阅的两页技术摘要；
- [ ] 明确哪些材料可以公开、哪些只适合私下分享。

在上述条件满足前，只准备联系人和材料模板，不发送尚未稳定的实验结论。

## 2. 建议建立的外部审阅组

### A. 无线通信研究者

重点审阅：

- UAV辅助WSN系统模型是否合理；
- 信道、吞吐、能耗和系统能效 \(\Xi\) 定义；
- Case 1/Case 2 是否足以支持泛化性；
- 与Sensors 2023原论文及近期工作的差异；
- 研究内容是否适合IEEE WCL。

### B. 强化学习研究者

重点审阅：

- NoDyna与DynaQ对照是否公平；
- Dyna模型和虚拟经验的使用是否严谨；
- K、warm-up和模型误差消融是否充分；
- 样本效率与墙钟成本的论证；
- 是否存在比固定K更合理的自适应规划方案。

### C. 统计与IEEE写作审阅者

重点审阅：

- 五种子配对检验是否恰当；
- 置信区间、效应量和多重比较；
- 是否存在选择性报告或过度结论；
- 五页Letter的信息取舍；
- 英文、符号、图表和IEEE格式。

## 3. 获取建议的渠道

按优先顺序执行：

1. IEEE Collabratec导师和技术社区；
2. IEEE ComSoc Young Professionals导师活动；
3. ComSoc相关技术委员会与邮件列表；
4. 直接联系相关论文的博士生、博士后或青年教师；
5. 与其他学校学生进行同行互审；
6. 稿件稳定后发布TechRxiv，并定向邀请反馈；
7. 必要时购买独立的技术审阅、统计审阅或英文编辑服务。

相关方向关键词：

- UAV communications；
- wireless sensor networks；
- model-based reinforcement learning；
- multi-agent reinforcement learning；
- green communications；
- resource allocation；
- IoT and sensor networks。

## 4. 联系对象筛选原则

优先联系：

- 近三年发表UAV通信与DRL论文的第一作者；
- MAHDRL、HRL-TPRA、HQMIX等相关工作的作者；
- 在无线通信中应用Dyna-Q或model-based RL的研究者；
- IEEE WCL近期UAV/DRL论文作者；
- 有明确公开导师意向的IEEE Collabratec成员。

避免：

- 与研究方向几乎无关但头衔很高的人；
- 要求先付费才能承诺“保证录用”的机构；
- 提供内部编辑关系、付费挂名或代投服务的人；
- 无法说明专业背景或保密方式的审阅者。

## 5. 首次联系材料

首次联系不发送完整论文，只准备一个两页审阅包：

1. 研究问题和系统模型摘要；
2. 一张算法架构图；
3. 一张核心三算法结果表；
4. 一张样本效率与墙钟成本图；
5. 当前创新点与最接近工作的区别；
6. 三个以内的明确问题；
7. 明确说明只请求简短建议，不默认邀请共同署名。

不得首次发送：

- 服务器账号、地址或密码；
- 未脱敏的运行日志；
- 可写入的代码仓库权限；
- 尚未归档的原始结果；
- 与咨询问题无关的整套工程文件。

## 6. 联系邮件模板

```text
Subject: Request for brief technical feedback on UAV model-based HRL work

Dear Dr. [Name],

I am studying joint UAV trajectory planning, transmission scheduling,
and access control in UAV-assisted wireless sensor networks.

Our main distinction is using a learned Dyna model in the lower discrete
controller to reduce real-environment interactions, while retaining a
hierarchical multi-agent upper controller.

Would you be willing to comment briefly on the following questions?

1. Is the distinction from existing hierarchical model-free MARL clear?
2. Is the Dyna-K/model-error ablation sufficient to support the claim?
3. Is the sample-efficiency versus wall-clock-cost discussion appropriate
   for an IEEE WCL Letter?

I can provide a two-page technical summary rather than the full manuscript.
Any brief feedback would be greatly appreciated.

Best regards,
[Name]
```

邮件应针对对方论文做个性化修改，禁止批量发送完全相同的内容。

## 7. 反馈记录表

每次咨询后记录：

| 字段 | 内容 |
|---|---|
| 联系人 | 姓名、机构、研究方向 |
| 联系日期 | YYYY-MM-DD |
| 渠道 | Collabratec / 邮件 / 会议 / 同行互审 |
| 分享材料 | 摘要、图表、私下稿件版本 |
| 核心意见 | 对创新、方法、实验或写作的反馈 |
| 是否采纳 | 是 / 部分 / 否 |
| 修改位置 | 论文或代码对应位置 |
| 署名判断 | 作者 / 致谢 / 无需列出 |
| 后续动作 | 补实验、改稿或再次确认 |

不同审阅者意见冲突时，不以头衔决定，优先依据技术理由、数据和WCL官方要求判断。

## 8. 保密与预印本边界

- 在公开论文前，优先发送两页摘要或只读PDF；
- 私下分享完整稿件时注明“confidential draft, not for redistribution”；
- 代码仓库只提供公开只读链接，不提供服务器访问权限；
- TechRxiv仅在方法、作者和主要结论稳定后发布；
- 发布预印本后按WCL电子发布政策处理投稿声明；
- 记录所有公开版本，防止不同奖励和实验口径混淆。

## 9. 署名与致谢规则

只有同时满足以下条件才讨论共同作者：

1. 对理论、方法、实验设计或数据解释作出实质性智力贡献；
2. 参与论文撰写或重要学术修订；
3. 审阅并批准最终投稿版本；
4. 愿意对论文内容承担作者责任。

仅提供一般建议、一次性审阅、语言修改或格式检查的人，原则上列入致谢，不进行挂名。
任何作者增加都必须在投稿前获得所有作者明确同意。

## 10. 执行顺序

### ER-0：现在

- [ ] 维护潜在联系人清单；
- [ ] 不发送未完成的正式消融结论；
- [ ] 保留本计划为待办状态。

### ER-1：阶段 D 完成后

- [ ] 生成正式消融报告；
- [ ] 更新两页技术摘要；
- [ ] 选择5–10名IEEE Collabratec导师或技术社区成员；
- [ ] 选择5名高度相关论文的青年作者；
- [ ] 准备个性化联系邮件。

### ER-2：第一轮审阅

- [ ] 获取至少一名无线通信研究者意见；
- [ ] 获取至少一名强化学习研究者意见；
- [ ] 整理意见冲突和必须补充的实验；
- [ ] 修改论文贡献定位与实验叙述。

### ER-3：投稿前模拟审稿

- [ ] 使用WCL标准完成一次完整技术审阅；
- [ ] 完成统计与图表终审；
- [ ] 完成英文和IEEE格式检查；
- [ ] 检查篇幅、摘要、索引词、引用和投稿声明；
- [ ] 决定是否达到投稿门槛。

## 11. 完成标准

投稿前至少满足：

- 一名通信方向审阅者确认系统模型无明显错误；
- 一名RL方向审阅者确认消融和创新表述基本成立；
- 统计结果与论文主张一致；
- 所有外部意见均有采纳记录或不采纳理由；
- 不存在不符合IEEE作者资格的挂名；
- 稿件能够在五页内完整说明问题、方法、实验和局限。

## 12. 当前下一步

当前不发送外部联系请求。继续完成阶段 D 正式消融；实验完成并归档后，先制作两页技术
摘要，再启动ER-1联系人筛选与定向联系。
