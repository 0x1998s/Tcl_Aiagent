"""
TCL AI Agent Python SDK 安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取版本
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "tcl_ai_agent_sdk", "__version__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            exec(f.read())
            return locals()["__version__"]
    return "1.0.0"

setup(
    name="tcl-ai-agent-sdk",
    version=read_version(),
    description="TCL AI Agent Python SDK - 企业级数据分析智能助手客户端",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Jemmy Yang",
    author_email="jemmy_yang@yeah.net",
    url="https://github.com/0x1998s/Tcl_Aiagent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Spreadsheet",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
        "websockets>=11.0.0",
        "dataclasses; python_version<'3.7'",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
        "pandas": [
            "pandas>=1.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tcl-ai-agent=tcl_ai_agent_sdk.cli:main",
        ],
    },
    keywords=[
        "ai", "agent", "data-analysis", "business-intelligence", 
        "tcl", "llm", "analytics", "dashboard", "reporting"
    ],
    project_urls={
        "Documentation": "https://github.com/0x1998s/Tcl_Aiagent#readme",
        "Source": "https://github.com/0x1998s/Tcl_Aiagent",
        "Tracker": "https://github.com/0x1998s/Tcl_Aiagent/issues",
    },
)
