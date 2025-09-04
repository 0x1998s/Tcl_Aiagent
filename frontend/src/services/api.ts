import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error);
    
    if (error.response?.status === 401) {
      // 处理认证失败
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

// 聊天API
export const chatAPI = {
  sendMessage: (data: {
    message: string;
    user_id?: string;
    session_id?: string;
    context?: any;
  }) => api.post('/chat', data),
  
  getHistory: (sessionId: string) => api.get(`/chat/history/${sessionId}`),
};

// 查询API
export const queryAPI = {
  execute: (data: {
    query: string;
    user_id?: string;
    context?: any;
    use_cache?: boolean;
  }) => api.post('/api/query/execute', data),
  
  getSchema: () => api.get('/api/query/schema'),
  
  getSampleData: (tableName: string, limit?: number) => 
    api.get(`/api/query/tables/${tableName}/sample`, { params: { limit } }),
  
  validateSQL: (sql: string) => api.post('/api/query/validate-sql', { sql }),
};

// 分析API
export const analysisAPI = {
  execute: (data: {
    analysis_type: string;
    metrics: string[];
    dimensions?: string[];
    filters?: any;
    time_range?: any;
    user_id?: string;
  }) => api.post('/api/analysis/execute', data),
  
  getMetrics: () => api.get('/api/analysis/metrics'),
  
  getAnalysisTypes: () => api.get('/api/analysis/analysis-types'),
};

// 预警API
export const alertAPI = {
  createRule: (data: {
    name: string;
    metric: string;
    condition: string;
    threshold: number;
    time_window: number;
    severity: string;
    enabled?: boolean;
    notification_channels?: string[];
  }) => api.post('/api/alerts/rules', data),
  
  getRules: () => api.get('/api/alerts/rules'),
  
  getActiveAlerts: () => api.get('/api/alerts/active'),
  
  checkMetrics: () => api.post('/api/alerts/check'),
  
  resolveAlert: (alertId: string) => api.post(`/api/alerts/alerts/${alertId}/resolve`),
  
  suppressAlert: (alertId: string, durationMinutes?: number) => 
    api.post(`/api/alerts/alerts/${alertId}/suppress`, { duration_minutes: durationMinutes }),
  
  getHistory: (params?: {
    start_date?: string;
    end_date?: string;
    severity?: string;
    limit?: number;
  }) => api.get('/api/alerts/history', { params }),
  
  getStats: () => api.get('/api/alerts/stats'),
};

// 报告API
export const reportAPI = {
  generate: (data: {
    report_type: string;
    title: string;
    metrics: string[];
    time_range: { start: string; end: string };
    format?: string;
    include_charts?: boolean;
    user_id?: string;
  }) => api.post('/api/reports/generate', data),
  
  getTemplates: () => api.get('/api/reports/templates'),
  
  getHistory: (userId?: string, limit?: number) => 
    api.get('/api/reports/history', { params: { user_id: userId, limit } }),
  
  getStatus: (reportId: string) => api.get(`/api/reports/${reportId}/status`),
  
  download: (reportId: string) => api.get(`/api/reports/${reportId}/download`),
  
  delete: (reportId: string) => api.delete(`/api/reports/${reportId}`),
  
  schedule: (data: {
    report_request: any;
    cron_expression: string;
    recipients: string[];
  }) => api.post('/api/reports/schedule', data),
};

// 实验API
export const experimentAPI = {
  create: (data: {
    name: string;
    description: string;
    type: string;
    hypothesis: string;
    success_metric: string;
    sample_size: number;
    confidence_level?: number;
    statistical_power?: number;
    variants: any[];
    created_by?: string;
  }) => api.post('/api/experiments/create', data),
  
  list: (status?: string, limit?: number) => 
    api.get('/api/experiments/list', { params: { status, limit } }),
  
  get: (experimentId: string) => api.get(`/api/experiments/${experimentId}`),
  
  start: (experimentId: string) => api.post(`/api/experiments/${experimentId}/start`),
  
  stop: (experimentId: string, reason?: string) => 
    api.post(`/api/experiments/${experimentId}/stop`, { reason }),
  
  getResults: (experimentId: string) => api.get(`/api/experiments/${experimentId}/results`),
  
  makeDecision: (experimentId: string, data: {
    decision: string;
    winner_variant?: string;
  }) => api.post(`/api/experiments/${experimentId}/decide`, data),
  
  calculateSampleSize: (params: {
    baseline_rate: number;
    minimum_effect: number;
    confidence_level?: number;
    statistical_power?: number;
  }) => api.get('/api/experiments/calculate/sample-size', { params }),
};

// 系统API
export const systemAPI = {
  getHealth: () => api.get('/health'),
  
  getMetrics: () => api.get('/metrics'),
  
  getStatus: () => api.get('/'),
};

export default api;
