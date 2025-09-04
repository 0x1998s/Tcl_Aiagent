-- TCL AI Agent 示例数据
-- 用于演示和测试

-- 用户表示例数据
INSERT INTO users (username, email, full_name, is_active, is_admin) VALUES
('admin', 'admin@tcl-ai-agent.demo', 'System Administrator', true, true),
('analyst', 'analyst@tcl-ai-agent.demo', 'Data Analyst', true, false),
('viewer', 'viewer@tcl-ai-agent.demo', 'Report Viewer', true, false);

-- 产品销售示例数据
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO products (name, category, price) VALUES
('TCL电视 55寸', '电视', 2999.00),
('TCL空调 1.5匹', '空调', 1899.00),
('TCL冰箱 双门', '冰箱', 2599.00),
('TCL洗衣机 8kg', '洗衣机', 1699.00);

-- 销售订单示例数据
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    product_id INTEGER,
    user_id INTEGER,
    quantity INTEGER,
    amount DECIMAL(10,2),
    order_date DATE,
    status VARCHAR(20),
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

INSERT INTO orders (product_id, user_id, quantity, amount, order_date, status) VALUES
(1, 2, 2, 5998.00, '2024-01-15', 'completed'),
(2, 3, 1, 1899.00, '2024-01-16', 'completed'),
(3, 2, 1, 2599.00, '2024-01-17', 'pending'),
(4, 3, 3, 5097.00, '2024-01-18', 'completed');
