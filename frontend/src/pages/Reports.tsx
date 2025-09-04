import React from 'react';
import { Typography, Card } from 'antd';

const { Title } = Typography;

const Reports: React.FC = () => {
  return (
    <div>
      <Title level={2}>报告生成</Title>
      <Card>
        <p>报告生成功能正在开发中...</p>
      </Card>
    </div>
  );
};

export default Reports;
