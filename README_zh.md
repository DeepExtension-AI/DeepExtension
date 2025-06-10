<div style="display: flex; align-items: center; justify-content: center;">
  <img src="docs/assets/favicon.svg" width="100" style="margin-right: 20px;" />
  <h1>DeepExtension</h1>
</div>

## 💡 1.DeepExtension是什么？

[DeepExtension](https://deepextension.ai/)是一个 AI 基础设施平台，帮助企业轻松管理大型语言模型（LLM）开发的完整生命周期——从数据准备到微调、评估和部署。

我们的使命是让面向特定领域的 AI 开发变得**易于使用、协作高效、易于扩展**，尤其适合缺乏 AI 专业人员或计算资源受限的团队。

无论您是 AI 工程师还是业务专家，DeepExtension 都为您提供一个协作环境，通过现代技术如 **PEFT**、**GRPO**，帮助构建高质量的模型——一切操作都通过模块化的网页界面完成。

## 📘 2.官方文档

请登录网址 [https://docs.deepextension.ai/zh/](https://docs.deepextension.ai/zh/) 查看官方文档。
<div align="left" style="margin-top:20px;margin-bottom:20px;">
<img src="docs/assets/image-20250528150548686.png" width="1000"/>
</div>

## 🎉 3.关注项目

⭐️ 点击右上角的 Star 关注 DeepExtension，可以获取最新发布的实时通知 !🌟

## 🚀 4. 快速开始

你可以在以下平台上安装 **DeepExtension**：

- ✅ **Linux** 或 **Windows（通过 WSL）** — 支持 **CUDA** 进行 GPU 训练  
- ✅ **macOS（Apple M 系列）** — 使用 **MLX** 后端  
- ✅ **任意 Linux/macOS 环境（非训练模式）** — 仅用于 UI 访问和推理

---

### 📝 前置条件

- **Docker 引擎**  
  如果尚未安装 Docker，请参考官方安装指南：  
  👉 [安装 Docker 引擎](https://docs.docker.com/engine/install/)

---

### 4.1 克隆代码仓库

```bash
git clone https://github.com/DeepExtension-AI/DeepExtension.git
cd DeepExtension
```

---

### 4.2 启动应用程序

运行启动脚本：

```bash
./run_compose.sh
```

请确保：

- 所需的 Docker 镜像已全部拉取
- 所有容器成功启动，无错误

---

#### 🎯 访问 Web UI

服务启动后，打开浏览器访问：  
[http://localhost:{webui_port}](http://localhost:{webui_port})

你可以在 `run_compose.sh` 的日志输出中或 Docker 引擎容器管理界面中找到 `{webui_port}`。

**示例：登录页面**

<div align="left" style="margin-top:20px;margin-bottom:20px;">
<img src="docs/assets/login.png" width="500"/>
</div>

---

#### 🔐 管理员首次登录

首次启动时会自动创建一个 **root 管理员用户**。

- **初始密码文件：**

  ```
  DeepExtension/adminPassword/adminPassword.txt
  ```

- **登录凭据：**

  ```
  项目代码： 1001
  用户名：   admin
  密码：     （见上方密码文件）
  ```

---

#### 🔧 当前可用功能

**DeepExtension** 当前支持：

- 管理第三方模型

- 对第三方模型执行推理和评估任务

- 通过 Web UI 监控输出结果和评估指标

---

### 4.3 配置模型训练环境

要启用 **本地模型** 的训练、评估、保存和部署功能，你需要配置训练环境。  
不同平台的配置步骤不同，请参考完整安装指南：  
👉 [配置模型训练环境](https://deepextension.readthedocs.io/en/latest/developer/install/#3-set-up-model-training-environment)

> **注意：**  
> 即使没有配置训练环境，你仍然可以使用第三方模型执行推理和评估任务。

---

### 📚 4.4 完整安装指南

如需高级配置（如基于 MLX 的训练或连接自定义数据库），请参考完整文档：  
👉 [安装指南](https://deepextension.readthedocs.io/en/latest/developer/install/)

## 🌟 5. 主要功能

- 🤖 **多模型对话与知识库引用**：支持与本地已训练模型及多种第三方模型（如主流AI平台）直接对话，结合个人或企业知识库，实现智能问答与知识检索。
- 🔌 **便捷集成第三方模型与平台**：可快速添加和切换多种第三方模型及平台，适用范围广，满足不同场景需求。
- 🚀 **本地模型训练与一键部署**：支持在本地环境下对模型进行训练与微调，训练成果可快速保存，并可在 Ollama 等环境下高效部署，极大提升模型迭代效率。
- 📚 **个人知识库快速构建**：支持一键创建个人知识库，灵活导入多种格式的文件（如PDF、DOCX、XLSX等），便于知识管理与扩展。
- 📊 **模型评估与对比分析**：内置模型评测工具，可对不同性能、不同版本的模型进行横向对比，帮助用户选择最适合的目标模型。
- 📊 **数据集上传与解析**：支持上传并自动解析模型训练所需的数据集，简化数据准备流程，加速模型开发。

## 📚 6. 技术文档

DeepExtension 采用模块化、分层式的系统架构，确保平台具备高可用性、可扩展性与易维护性。整体架构设计如下：

### 🏗️ 系统架构概览

```
┌─────────────────────────────┐
          Web 前端界面         
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
         后端服务（API 层）     
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
    任务编排 & 管理（TaskFlow）  
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
     模型服务（推理/训练/评估）    
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
      数据与知识库（Data/KB）      
└─────────────────────────────┘
```

### 主要模块说明

- **Web 前端界面**：提供直观易用的操作界面，支持模型管理、知识库管理、任务编排、数据集上传等全流程操作。
- **后端服务（API 层）**：负责业务逻辑处理、权限校验、任务调度等，连接前端与底层服务。
- **任务编排与管理（TaskFlow）**：实现模型训练、评估、推理等任务的自动化编排与流程管理，支持多任务并发与状态追踪。
- **模型服务**：支持本地及第三方模型的推理、训练与评估，具备灵活的模型注册与调用机制。
- **数据与知识库**：统一管理结构化与非结构化数据，支持多格式文件导入、知识抽取与检索。

### 架构优势

- **模块解耦，灵活扩展**：各模块独立部署，便于功能扩展与维护。
- **高可用与弹性伸缩**：支持分布式部署与负载均衡，满足不同规模需求。
- **安全合规**：支持细粒度权限控制与数据隔离，保障数据安全。
- **开放集成**：支持丰富调用三方API，便于与外部第三方模型平台对接。

> 详细架构介绍请参考[官方架构文档](https://docs.deepextension.ai/zh/intro/architecture/)。

- [安装指南](https://docs.deepextension.ai/zh/developer/install/)
- [常见问题 FAQs](https://docs.deepextension.ai/zh/faq/)
- [架构文档](https://docs.deepextension.ai/zh/intro/architecture/)
## 🏄 7. 开源社区
- 文档中心：访问官方 DeepExtension 文档门户，[文档地址:https://docs.deepextension.ai/zh/](https://docs.deepextension.ai/zh/)
- 社区论坛：参与讨论、提交反馈或提出功能建议（即将上线）
- GitHub：关注版本发布、提交问题、参与我们的开源组件项目，[项目地址:https://github.com/DeepExtension-AI/DeepExtension](https://github.com/DeepExtension-AI/DeepExtension)
## 🙌 8. 技术支持
如果在使用 DeepExtension 时遇到问题：
1. 您可以优先查阅相关文档与常见问题解答；
2. 若问题仍未解决，你也可以通过邮件support@deepextension.ai联系我们，并提供以下信息：
- DeepExtension 的版本号
- 所使用的操作系统 / 环境
- 详细错误信息或截图（如适用）
我们将在两个工作日内给予回复。
## 🤝 9. 联系我们
如有一般性咨询、合作或媒体采访需求，请通过以下方式联系我们：

- 邮箱：contact@deepextension.ai
- 官网：https://www.deepextension.ai
- 微信公众号: 简帷软件
<div align="left" style="margin-top:20px;margin-bottom:20px; ">
<img src="docs/assets/wechat.png" width="300"/>
</div>

## 👥 10. 意见与功能建议
我们非常重视您的反馈，欢迎告知好用的功能和存在的问题，以及您希望未来能加入功能。
反馈渠道：

- 邮箱：contact@deepextension.ai
- GitHub Issues（适用于技术相关建议）


感谢您对 DeepExtension 的支持！🚀