import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const Analytics: React.FC = () => {
  return (
    <div>
      <Title level={2}>数据分析</Title>
      <Card>
        <p>数据分析功能正在开发中...</p>
      </Card>
    </div>
  );
};

export default Analytics;
