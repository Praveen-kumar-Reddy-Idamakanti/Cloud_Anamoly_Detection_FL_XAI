
import {
  mockStats,
  mockAnomalies,
  mockLogs,
  mockExplanations,
  timeSeriesData,
  AnomalyData,
  ExplanationData,
  LogData,
  StatData,
  TimeSeriesData,
  generateRealtimeData
} from '../data/mockData';

// Simulate API latency
const simulateDelay = () => new Promise((resolve) => setTimeout(resolve, Math.random() * 500 + 300));

// Simulate error (approximately 5% of the time)
const shouldError = () => Math.random() < 0.05;

export type LoginCredentials = {
  email: string;
  password: string;
};

export type RegisterData = {
  name: string;
  email: string;
  password: string;
};

export type ApiError = {
  message: string;
  code: number;
};

export type AuthResponse = {
  token: string;
  user: {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'user';
  };
};

// Authentication API
export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Authentication failed. Please check your credentials.', code: 401 };
    }
    
    if (credentials.email && credentials.password) {
      return {
        token: `jwt-mock-token-${Date.now()}`,
        user: {
          id: '1',
          name: credentials.email.split('@')[0],
          email: credentials.email,
          role: Math.random() > 0.7 ? 'admin' : 'user',
        },
      };
    }
    
    throw { message: 'Email and password are required', code: 400 };
  },
  
  register: async (data: RegisterData): Promise<AuthResponse> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Registration failed. Please try again later.', code: 500 };
    }
    
    if (data.email && data.password && data.name) {
      return {
        token: `jwt-mock-token-${Date.now()}`,
        user: {
          id: '1',
          name: data.name,
          email: data.email,
          role: 'user',
        },
      };
    }
    
    throw { message: 'All fields are required', code: 400 };
  },
  
  logout: async (): Promise<void> => {
    await simulateDelay();
    return;
  },
};

// Stats API
export const statsApi = {
  getStats: async (): Promise<StatData> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch statistics', code: 500 };
    }
    
    return mockStats;
  },
  
  getTimeSeriesData: async (days: number = 30): Promise<TimeSeriesData[]> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch time series data', code: 500 };
    }
    
    return timeSeriesData.slice(-days);
  },
};

// Anomalies API
export const anomaliesApi = {
  getAnomalies: async (page: number = 1, limit: number = 10): Promise<{ data: AnomalyData[], total: number }> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch anomalies', code: 500 };
    }
    
    const start = (page - 1) * limit;
    const end = start + limit;
    const paginatedData = mockAnomalies.slice(start, end);
    
    return {
      data: paginatedData,
      total: mockAnomalies.length,
    };
  },
  
  getAnomalyById: async (id: string): Promise<AnomalyData> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch anomaly details', code: 500 };
    }
    
    const anomaly = mockAnomalies.find(a => a.id === id);
    
    if (!anomaly) {
      throw { message: 'Anomaly not found', code: 404 };
    }
    
    return anomaly;
  },
  
  reviewAnomaly: async (id: string, reviewed: boolean): Promise<AnomalyData> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to update anomaly review status', code: 500 };
    }
    
    const anomalyIndex = mockAnomalies.findIndex(a => a.id === id);
    
    if (anomalyIndex === -1) {
      throw { message: 'Anomaly not found', code: 404 };
    }
    
    const updatedAnomaly = { ...mockAnomalies[anomalyIndex], reviewed };
    mockAnomalies[anomalyIndex] = updatedAnomaly;
    
    return updatedAnomaly;
  },
};

// Logs API
export const logsApi = {
  getLogs: async (page: number = 1, limit: number = 10): Promise<{ data: LogData[], total: number }> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch logs', code: 500 };
    }
    
    const start = (page - 1) * limit;
    const end = start + limit;
    const paginatedData = mockLogs.slice(start, end);
    
    return {
      data: paginatedData,
      total: mockLogs.length,
    };
  },
  
  uploadLog: async (file: File, encrypted: boolean): Promise<LogData> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to upload log', code: 500 };
    }
    
    // Generate a mock log entry
    const newLog: LogData = {
      id: `log-${Date.now()}`,
      timestamp: new Date().toISOString(),
      sourceIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      destinationIp: `10.0.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      protocol: ['TCP', 'UDP', 'HTTP', 'HTTPS'][Math.floor(Math.random() * 4)],
      encrypted,
      size: file.size,
    };
    
    mockLogs.unshift(newLog);
    
    return newLog;
  },
};

// XAI Explanations API
export const explanationsApi = {
  getExplanation: async (anomalyId: string): Promise<ExplanationData> => {
    await simulateDelay();
    
    if (shouldError()) {
      throw { message: 'Failed to fetch explanation', code: 500 };
    }
    
    const explanation = mockExplanations.find(e => e.anomalyId === anomalyId);
    
    if (!explanation) {
      throw { message: 'Explanation not found', code: 404 };
    }
    
    return explanation;
  },
};

// Real-time data simulation
let eventSource: null | (() => void) = null;

export const realtimeApi = {
  subscribeToRealTimeUpdates: (callback: (data: { log?: LogData, anomaly?: AnomalyData }) => void) => {
    // Cleanup any existing subscription
    if (eventSource) {
      realtimeApi.unsubscribeFromRealTimeUpdates();
    }
    
    // Create a new interval for sending updates
    const interval = setInterval(() => {
      const { newLog, newAnomaly } = generateRealtimeData();
      
      // Add to mock data
      if (newLog) {
        mockLogs.unshift(newLog);
      }
      
      if (newAnomaly) {
        mockAnomalies.unshift(newAnomaly);
        
        // Create an explanation for this anomaly
        const newExplanation: ExplanationData = {
          id: `explanation-${newAnomaly.id}`,
          anomalyId: newAnomaly.id,
          modelType: ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM'][Math.floor(Math.random() * 3)] as ExplanationData['modelType'],
          shap: [
            { feature: 'packet_size', importance: Math.random() * 0.5 },
            { feature: 'protocol', importance: Math.random() * 0.5 },
            { feature: 'time_of_day', importance: Math.random() * 0.5 },
            { feature: 'source_ip_reputation', importance: Math.random() * 0.5 },
            { feature: 'connection_frequency', importance: Math.random() * 0.5 },
          ].sort((a, b) => b.importance - a.importance),
          lime: [
            { feature: 'packet_size', importance: Math.random() * 0.5 },
            { feature: 'protocol', importance: Math.random() * 0.5 },
            { feature: 'time_of_day', importance: Math.random() * 0.5 },
            { feature: 'source_ip_reputation', importance: Math.random() * 0.5 },
            { feature: 'connection_frequency', importance: Math.random() * 0.5 },
          ].sort((a, b) => b.importance - a.importance),
          contributingFactors: [
            'Unusual time of access',
            'Connection from untrusted IP range',
            'Abnormal data transfer volume',
            'Suspicious protocol usage',
            'Deviation from baseline behavior'
          ].slice(0, Math.floor(Math.random() * 3) + 2),
          recommendations: [
            'Monitor source IP for additional suspicious activity',
            'Verify legitimacy of data transfers',
            'Apply additional authentication for this source',
            'Block access from this IP range',
            'Update firewall rules to restrict this type of traffic'
          ].slice(0, Math.floor(Math.random() * 3) + 1),
        };
        
        mockExplanations.unshift(newExplanation);
      }
      
      // Send to callback
      callback({ 
        log: newLog,
        anomaly: newAnomaly || undefined 
      });
    }, 5000);
    
    // Store the cleanup function
    eventSource = () => clearInterval(interval);
    
    return () => {
      if (eventSource) {
        eventSource();
        eventSource = null;
      }
    };
  },
  
  unsubscribeFromRealTimeUpdates: () => {
    if (eventSource) {
      eventSource();
      eventSource = null;
    }
  },
};
