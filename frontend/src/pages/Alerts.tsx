import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const Alerts: React.FC = () => {
  return (
    <div>
      <Title level={2}>预警监控</Title>
      <Card>
        <p>预警监控功能正在开发中...</p>
      </Card>
    </div>
  );
};

export default Alerts;
