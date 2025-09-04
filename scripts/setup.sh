#!/bin/bash

# TCL AI Agent 项目设置脚本

set -e

echo "🚀 开始设置 TCL AI Agent 项目..."

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p logs data/uploads reports templates

# 检查Python版本
echo "🐍 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ 需要Python 3.9或更高版本"
    exit 1
fi

# 检查Node.js版本
echo "📦 检查Node.js版本..."
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "Node.js版本: $node_version"
else
    echo "❌ 未找到Node.js，请先安装Node.js 16+"
    exit 1
fi

# 设置Python虚拟环境
echo "🔧 设置Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 创建虚拟环境成功"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装Python依赖
echo "📦 安装Python依赖..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Python依赖安装完成"

cd ..

# 安装前端依赖
echo "📦 安装前端依赖..."
cd frontend
npm install
echo "✅ 前端依赖安装完成"

cd ..

# 复制配置文件
echo "⚙️  设置配置文件..."
if [ ! -f ".env" ]; then
    cp configs/env.example .env
    echo "✅ 已创建.env配置文件，请根据需要修改"
else
    echo "ℹ️  .env文件已存在，跳过创建"
fi

# 检查Docker
echo "🐳 检查Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker已安装"
    if command -v docker-compose &> /dev/null; then
        echo "✅ Docker Compose已安装"
    else
        echo "⚠️  Docker Compose未安装，建议安装以便使用容器部署"
    fi
else
    echo "⚠️  Docker未安装，建议安装以便使用容器部署"
fi

# 创建示例数据
echo "📊 准备示例数据..."
cat > data/init.sql << EOF
-- 创建示例数据库表和数据
-- 这个文件将在PostgreSQL容器启动时执行

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    age INTEGER,
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- 产品表
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock INTEGER DEFAULT 0
);

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- 插入示例数据
INSERT INTO users (username, email, age, city, created_at, last_login) VALUES
('张三', 'zhangsan@example.com', 25, '北京', '2024-01-01', '2024-01-15'),
('李四', 'lisi@example.com', 30, '上海', '2024-01-02', '2024-01-14'),
('王五', 'wangwu@example.com', 28, '广州', '2024-01-03', '2024-01-13')
ON CONFLICT (email) DO NOTHING;

INSERT INTO products (product_name, category, price, stock) VALUES
('iPhone 15', '手机', 7999.00, 100),
('MacBook Pro', '电脑', 15999.00, 50),
('iPad Air', '平板', 4599.00, 80)
ON CONFLICT DO NOTHING;

INSERT INTO orders (user_id, product_id, quantity, amount, order_date, status) VALUES
(1, 1, 1, 7999.00, '2024-01-10', 'completed'),
(2, 2, 1, 15999.00, '2024-01-11', 'completed'),
(3, 3, 2, 9198.00, '2024-01-12', 'pending')
ON CONFLICT DO NOTHING;
EOF

echo "✅ 示例数据文件已创建"

# 创建启动脚本
echo "🚀 创建启动脚本..."
cat > scripts/start.sh << EOF
#!/bin/bash

# 启动TCL AI Agent项目

echo "🚀 启动TCL AI Agent..."

# 检查是否在虚拟环境中
if [[ "\$VIRTUAL_ENV" == "" ]]; then
    echo "激活Python虚拟环境..."
    source venv/bin/activate
fi

# 启动后端服务
echo "🖥️  启动后端服务..."
cd backend
python main.py &
BACKEND_PID=\$!

cd ..

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 5

# 启动前端服务
echo "🌐 启动前端服务..."
cd frontend
npm start &
FRONTEND_PID=\$!

cd ..

echo "✅ 服务启动完成！"
echo "📊 前端地址: http://localhost:3000"
echo "🔧 后端API: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"

# 等待用户输入以停止服务
echo "按 Ctrl+C 停止服务..."
trap "echo '🛑 停止服务...'; kill \$BACKEND_PID \$FRONTEND_PID; exit" INT
wait
EOF

chmod +x scripts/start.sh

cat > scripts/start-docker.sh << EOF
#!/bin/bash

# 使用Docker启动TCL AI Agent项目

echo "🐳 使用Docker启动TCL AI Agent..."

# 构建和启动服务
docker-compose up --build -d

echo "✅ Docker服务启动完成！"
echo "📊 前端地址: http://localhost:3000"
echo "🔧 后端API: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: docker-compose down"
EOF

chmod +x scripts/start-docker.sh

echo "✅ 启动脚本已创建"

echo ""
echo "🎉 项目设置完成！"
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，配置必要的环境变量（如OpenAI API Key等）"
echo "2. 选择启动方式："
echo "   - 本地启动: ./scripts/start.sh"
echo "   - Docker启动: ./scripts/start-docker.sh"
echo ""
echo "📚 更多信息请查看 README.md"
echo ""
