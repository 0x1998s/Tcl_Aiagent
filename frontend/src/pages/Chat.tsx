import React, { useState, useRef, useEffect } from 'react';
import {
  Input,
  Button,
  Card,
  Space,
  Avatar,
  Typography,
  Spin,
  message,
  Row,
  Col,
  Tag,
  Divider,
  Empty
} from 'antd';
import {
  SendOutlined,
  UserOutlined,
  RobotOutlined,
  ClearOutlined,
  HistoryOutlined
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { chatAPI } from '../services/api';

const { TextArea } = Input;
const { Text, Title } = Typography;

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  data?: any;
  charts?: any[];
}

interface ChatProps {}

const Chat: React.FC<ChatProps> = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 发送消息
  const handleSend = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await chatAPI.sendMessage({
        message: inputValue.trim(),
        user_id: 'default',
        session_id: 'default'
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response || '抱歉，我没有理解您的问题。',
        timestamp: new Date(),
        data: response.data,
        charts: response.charts
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('发送消息失败:', error);
      message.error('发送消息失败，请重试');
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: '抱歉，服务暂时不可用，请稍后重试。',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // 清空对话
  const handleClear = () => {
    setMessages([]);
  };

  // 处理回车发送
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 渲染图表
  const renderChart = (chart: any) => {
    const option = {
      title: {
        text: chart.title || '数据图表',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        top: 30
      },
      xAxis: {
        type: 'category',
        data: chart.data?.map((item: any) => item[chart.x_axis || 'date']) || []
      },
      yAxis: {
        type: 'value'
      },
      series: [{
        name: chart.title || '数据',
        type: chart.type === 'line' ? 'line' : 'bar',
        data: chart.data?.map((item: any) => item[chart.y_axis || 'value']) || [],
        smooth: chart.type === 'line'
      }]
    };

    return (
      <ReactECharts 
        option={option} 
        style={{ height: '300px', width: '100%' }}
      />
    );
  };

  // 渲染数据表格
  const renderDataTable = (data: any[]) => {
    if (!data || data.length === 0) return null;

    const keys = Object.keys(data[0]);
    
    return (
      <div style={{ 
        maxHeight: '200px', 
        overflowY: 'auto',
        border: '1px solid #f0f0f0',
        borderRadius: '6px',
        marginTop: '12px'
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ backgroundColor: '#fafafa' }}>
            <tr>
              {keys.map(key => (
                <th key={key} style={{ 
                  padding: '8px 12px', 
                  borderBottom: '1px solid #f0f0f0',
                  textAlign: 'left',
                  fontSize: '12px'
                }}>
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 10).map((row, index) => (
              <tr key={index}>
                {keys.map(key => (
                  <td key={key} style={{ 
                    padding: '8px 12px', 
                    borderBottom: '1px solid #f0f0f0',
                    fontSize: '12px'
                  }}>
                    {typeof row[key] === 'number' ? 
                      row[key].toLocaleString() : 
                      String(row[key])
                    }
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 10 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '8px',
            color: '#999',
            fontSize: '12px'
          }}>
            显示前10条，共{data.length}条数据
          </div>
        )}
      </div>
    );
  };

  // 渲染消息
  const renderMessage = (msg: Message) => {
    const isUser = msg.type === 'user';
    
    return (
      <div key={msg.id} style={{ 
        display: 'flex', 
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '16px'
      }}>
        <div style={{ 
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          maxWidth: '80%'
        }}>
          <Avatar 
            icon={isUser ? <UserOutlined /> : <RobotOutlined />}
            style={{ 
              backgroundColor: isUser ? '#1890ff' : '#52c41a',
              marginLeft: isUser ? '8px' : '0',
              marginRight: isUser ? '0' : '8px'
            }}
          />
          
          <Card 
            size="small"
            style={{
              backgroundColor: isUser ? '#e6f7ff' : '#f6ffed',
              border: `1px solid ${isUser ? '#91d5ff' : '#b7eb8f'}`,
              borderRadius: '12px',
              maxWidth: '600px'
            }}
          >
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {msg.content}
            </div>
            
            {/* 渲染图表 */}
            {msg.charts && msg.charts.length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <Divider orientation="left" style={{ fontSize: '12px' }}>
                  数据可视化
                </Divider>
                {msg.charts.map((chart, index) => (
                  <div key={index} style={{ marginBottom: '16px' }}>
                    {renderChart(chart)}
                  </div>
                ))}
              </div>
            )}
            
            {/* 渲染数据表格 */}
            {msg.data && Array.isArray(msg.data) && msg.data.length > 0 && (
              <div>
                <Divider orientation="left" style={{ fontSize: '12px' }}>
                  数据详情
                </Divider>
                {renderDataTable(msg.data)}
              </div>
            )}
            
            <div style={{ 
              marginTop: '8px', 
              fontSize: '11px', 
              color: '#999',
              textAlign: 'right'
            }}>
              {msg.timestamp.toLocaleTimeString()}
            </div>
          </Card>
        </div>
      </div>
    );
  };

  return (
    <div style={{ height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
      <Row gutter={16} style={{ marginBottom: '16px' }}>
        <Col span={18}>
          <Title level={3} style={{ margin: 0 }}>
            智能数据分析对话
          </Title>
        </Col>
        <Col span={6} style={{ textAlign: 'right' }}>
          <Space>
            <Button 
              icon={<HistoryOutlined />} 
              onClick={() => message.info('对话历史功能开发中')}
            >
              历史记录
            </Button>
            <Button 
              icon={<ClearOutlined />} 
              onClick={handleClear}
              disabled={messages.length === 0}
            >
              清空对话
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 消息区域 */}
      <div style={{ 
        flex: 1, 
        overflowY: 'auto',
        padding: '16px',
        backgroundColor: '#fafafa',
        borderRadius: '8px',
        marginBottom: '16px'
      }}>
        {messages.length === 0 ? (
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center' 
          }}>
            <Empty 
              description="开始您的数据分析对话吧！"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <div style={{ marginTop: '16px' }}>
                <Text type="secondary">您可以询问：</Text>
                <div style={{ marginTop: '8px' }}>
                  <Tag color="blue" style={{ margin: '2px' }}>今天的销售额是多少？</Tag>
                  <Tag color="green" style={{ margin: '2px' }}>分析用户增长趋势</Tag>
                  <Tag color="orange" style={{ margin: '2px' }}>生成本月销售报告</Tag>
                </div>
              </div>
            </Empty>
          </div>
        ) : (
          <>
            {messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </>
        )}
        
        {loading && (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'flex-start',
            marginBottom: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start' }}>
              <Avatar 
                icon={<RobotOutlined />}
                style={{ backgroundColor: '#52c41a', marginRight: '8px' }}
              />
              <Card 
                size="small"
                style={{
                  backgroundColor: '#f6ffed',
                  border: '1px solid #b7eb8f',
                  borderRadius: '12px'
                }}
              >
                <Spin size="small" />
                <Text style={{ marginLeft: '8px' }}>AI正在思考中...</Text>
              </Card>
            </div>
          </div>
        )}
      </div>

      {/* 输入区域 */}
      <Card size="small">
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px' }}>
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="请输入您的问题... (Shift+Enter换行，Enter发送)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            style={{ flex: 1 }}
            disabled={loading}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            loading={loading}
            disabled={!inputValue.trim()}
          >
            发送
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default Chat;
