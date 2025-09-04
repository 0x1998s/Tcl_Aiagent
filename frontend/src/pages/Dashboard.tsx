import React, { useState, useEffect } from 'react';
import {
  Row,
  Col,
  Card,
  Statistic,
  Table,
  Badge,
  Typography,
  Space,
  Button,
  Alert,
  Spin,
  message
} from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  DollarOutlined,
  ShoppingCartOutlined,
  UserOutlined,
  BellOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { systemAPI, alertAPI } from '../services/api';

const { Title, Text } = Typography;

interface DashboardData {
  metrics: {
    sales: { value: number; change: number };
    orders: { value: number; change: number };
    users: { value: number; change: number };
    conversion: { value: number; change: number };
  };
  charts: {
    salesTrend: any[];
    userGrowth: any[];
  };
  alerts: any[];
  recentReports: any[];
}

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<DashboardData | null>(null);

  // 获取仪表盘数据
  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      // 模拟获取数据
      const mockData: DashboardData = {
        metrics: {
          sales: { value: 125680, change: 12.5 },
          orders: { value: 1234, change: -3.2 },
          users: { value: 8765, change: 8.1 },
          conversion: { value: 3.45, change: 0.8 }
        },
        charts: {
          salesTrend: [
            { date: '01-01', value: 8000 },
            { date: '01-02', value: 9200 },
            { date: '01-03', value: 8800 },
            { date: '01-04', value: 10500 },
            { date: '01-05', value: 11200 },
            { date: '01-06', value: 9800 },
            { date: '01-07', value: 12500 }
          ],
          userGrowth: [
            { date: '01-01', value: 1200 },
            { date: '01-02', value: 1350 },
            { date: '01-03', value: 1280 },
            { date: '01-04', value: 1450 },
            { date: '01-05', value: 1520 },
            { date: '01-06', value: 1380 },
            { date: '01-07', value: 1600 }
          ]
        },
        alerts: [
          {
            id: 'alert_1',
            message: '日销售额下降5.6%，低于预警阈值',
            severity: 'medium',
            created_at: '2024-01-15T10:30:00'
          },
          {
            id: 'alert_2',
            message: '活跃用户数850，低于预警阈值1000',
            severity: 'high',
            created_at: '2024-01-15T09:15:00'
          }
        ],
        recentReports: [
          {
            id: 'report_1',
            title: '日销售报告',
            type: 'daily',
            generated_at: '2024-01-15T08:00:00',
            status: 'completed'
          },
          {
            id: 'report_2',
            title: '周度用户分析',
            type: 'weekly',
            generated_at: '2024-01-14T18:00:00',
            status: 'completed'
          }
        ]
      };

      setData(mockData);
    } catch (error) {
      console.error('获取仪表盘数据失败:', error);
      message.error('获取数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  // 销售趋势图表配置
  const getSalesTrendOption = () => {
    if (!data) return {};

    return {
      title: {
        text: '近7天销售趋势',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: '{b}<br/>销售额: ¥{c}'
      },
      xAxis: {
        type: 'category',
        data: data.charts.salesTrend.map(item => item.date)
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '¥{value}'
        }
      },
      series: [{
        data: data.charts.salesTrend.map(item => item.value),
        type: 'line',
        smooth: true,
        areaStyle: {
          opacity: 0.3
        },
        itemStyle: {
          color: '#1890ff'
        }
      }]
    };
  };

  // 用户增长图表配置
  const getUserGrowthOption = () => {
    if (!data) return {};

    return {
      title: {
        text: '近7天用户增长',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        formatter: '{b}<br/>用户数: {c}'
      },
      xAxis: {
        type: 'category',
        data: data.charts.userGrowth.map(item => item.date)
      },
      yAxis: {
        type: 'value'
      },
      series: [{
        data: data.charts.userGrowth.map(item => item.value),
        type: 'bar',
        itemStyle: {
          color: '#52c41a'
        }
      }]
    };
  };

  // 预警表格列配置
  const alertColumns = [
    {
      title: '预警信息',
      dataIndex: 'message',
      key: 'message',
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => {
        const colors = {
          low: 'default',
          medium: 'warning',
          high: 'error',
          critical: 'error'
        };
        return <Badge status={colors[severity as keyof typeof colors] as any} text={severity} />;
      }
    },
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ];

  // 报告表格列配置
  const reportColumns = [
    {
      title: '报告名称',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Badge color="blue" text={type} />
    },
    {
      title: '生成时间',
      dataIndex: 'generated_at',
      key: 'generated_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={status === 'completed' ? 'success' : 'processing'} 
          text={status === 'completed' ? '已完成' : '处理中'} 
        />
      )
    }
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: '16px' }}>
          <Text>加载仪表盘数据中...</Text>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <Alert
        message="数据加载失败"
        description="无法获取仪表盘数据，请刷新页面重试"
        type="error"
        showIcon
        action={
          <Button size="small" onClick={fetchDashboardData}>
            重新加载
          </Button>
        }
      />
    );
  }

  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
        <Col>
          <Title level={2} style={{ margin: 0 }}>
            数据概览仪表盘
          </Title>
          <Text type="secondary">实时监控业务关键指标</Text>
        </Col>
        <Col>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={fetchDashboardData}
            loading={loading}
          >
            刷新数据
          </Button>
        </Col>
      </Row>

      {/* 核心指标卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总销售额"
              value={data.metrics.sales.value}
              precision={0}
              valueStyle={{ color: data.metrics.sales.change > 0 ? '#3f8600' : '#cf1322' }}
              prefix={<DollarOutlined />}
              suffix={
                <span style={{ fontSize: '14px' }}>
                  {data.metrics.sales.change > 0 ? (
                    <ArrowUpOutlined style={{ color: '#3f8600' }} />
                  ) : (
                    <ArrowDownOutlined style={{ color: '#cf1322' }} />
                  )}
                  {Math.abs(data.metrics.sales.change)}%
                </span>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="订单数量"
              value={data.metrics.orders.value}
              valueStyle={{ color: data.metrics.orders.change > 0 ? '#3f8600' : '#cf1322' }}
              prefix={<ShoppingCartOutlined />}
              suffix={
                <span style={{ fontSize: '14px' }}>
                  {data.metrics.orders.change > 0 ? (
                    <ArrowUpOutlined style={{ color: '#3f8600' }} />
                  ) : (
                    <ArrowDownOutlined style={{ color: '#cf1322' }} />
                  )}
                  {Math.abs(data.metrics.orders.change)}%
                </span>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={data.metrics.users.value}
              valueStyle={{ color: data.metrics.users.change > 0 ? '#3f8600' : '#cf1322' }}
              prefix={<UserOutlined />}
              suffix={
                <span style={{ fontSize: '14px' }}>
                  {data.metrics.users.change > 0 ? (
                    <ArrowUpOutlined style={{ color: '#3f8600' }} />
                  ) : (
                    <ArrowDownOutlined style={{ color: '#cf1322' }} />
                  )}
                  {Math.abs(data.metrics.users.change)}%
                </span>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="转化率"
              value={data.metrics.conversion.value}
              precision={2}
              valueStyle={{ color: data.metrics.conversion.change > 0 ? '#3f8600' : '#cf1322' }}
              suffix={
                <span style={{ fontSize: '14px' }}>
                  %
                  {data.metrics.conversion.change > 0 ? (
                    <ArrowUpOutlined style={{ color: '#3f8600', marginLeft: '4px' }} />
                  ) : (
                    <ArrowDownOutlined style={{ color: '#cf1322', marginLeft: '4px' }} />
                  )}
                  {Math.abs(data.metrics.conversion.change)}%
                </span>
              }
            />
          </Card>
        </Col>
      </Row>

      {/* 图表区域 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={12}>
          <Card title="销售趋势" size="small">
            <ReactECharts 
              option={getSalesTrendOption()} 
              style={{ height: '300px' }}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="用户增长" size="small">
            <ReactECharts 
              option={getUserGrowthOption()} 
              style={{ height: '300px' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 预警和报告 */}
      <Row gutter={16}>
        <Col span={12}>
          <Card 
            title={
              <Space>
                <BellOutlined />
                活跃预警
                <Badge count={data.alerts.length} />
              </Space>
            }
            size="small"
          >
            <Table
              dataSource={data.alerts}
              columns={alertColumns}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ y: 200 }}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="最近报告" size="small">
            <Table
              dataSource={data.recentReports}
              columns={reportColumns}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ y: 200 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
