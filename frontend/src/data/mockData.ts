
export type AnomalyData = {
  id: string;
  timestamp: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  sourceIp: string;
  destinationIp: string;
  protocol: string;
  action: string;
  confidence: number;
  reviewed: boolean;
  details: string;
};

export type LogData = {
  id: string;
  timestamp: string;
  sourceIp: string;
  destinationIp: string;
  protocol: string;
  encrypted: boolean;
  size: number;
};

export type StatData = {
  totalLogs: number;
  totalAnomalies: number;
  criticalAnomalies: number;
  highAnomalies: number;
  mediumAnomalies: number;
  lowAnomalies: number;
  alertRate: number;
  avgConfidence: number;
};

export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type ExplanationData = {
  id: string;
  anomalyId: string;
  modelType: 'Isolation Forest' | 'Local Outlier Factor' | 'One-Class SVM';
  shap: FeatureImportance[];
  lime: FeatureImportance[];
  contributingFactors: string[];
  recommendations: string[];
};

export type TimeSeriesData = {
  date: string;
  logs: number;
  anomalies: number;
};

// Generate random IP addresses
const generateIp = () => {
  return `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(
    Math.random() * 255
  )}.${Math.floor(Math.random() * 255)}`;
};

// Generate mock anomalies
export const mockAnomalies: AnomalyData[] = Array.from({ length: 50 }, (_, i) => ({
  id: `anomaly-${i + 1}`,
  timestamp: new Date(Date.now() - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000)).toISOString(),
  severity: ['critical', 'high', 'medium', 'low'][Math.floor(Math.random() * 4)] as AnomalyData['severity'],
  sourceIp: generateIp(),
  destinationIp: generateIp(),
  protocol: ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH'][Math.floor(Math.random() * 6)],
  action: ['blocked', 'allowed', 'flagged', 'quarantined'][Math.floor(Math.random() * 4)],
  confidence: Math.round(Math.random() * 100) / 100,
  reviewed: Math.random() > 0.7,
  details: [
    'Unusual traffic pattern detected between hosts',
    'Multiple failed login attempts',
    'Suspicious outbound data transfer',
    'Port scanning activity',
    'Potential data exfiltration',
    'Unusual API access pattern',
    'Credential stuffing attempt',
    'Brute force attack detected',
  ][Math.floor(Math.random() * 8)],
})).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

// Generate mock logs
export const mockLogs: LogData[] = Array.from({ length: 200 }, (_, i) => ({
  id: `log-${i + 1}`,
  timestamp: new Date(Date.now() - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000)).toISOString(),
  sourceIp: generateIp(),
  destinationIp: generateIp(),
  protocol: ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH'][Math.floor(Math.random() * 6)],
  encrypted: Math.random() > 0.3,
  size: Math.floor(Math.random() * 5000) + 500,
}));

// Generate mock stats
export const mockStats: StatData = {
  totalLogs: mockLogs.length,
  totalAnomalies: mockAnomalies.length,
  criticalAnomalies: mockAnomalies.filter(a => a.severity === 'critical').length,
  highAnomalies: mockAnomalies.filter(a => a.severity === 'high').length,
  mediumAnomalies: mockAnomalies.filter(a => a.severity === 'medium').length,
  lowAnomalies: mockAnomalies.filter(a => a.severity === 'low').length,
  alertRate: parseFloat((mockAnomalies.length / mockLogs.length * 100).toFixed(2)),
  avgConfidence: parseFloat((mockAnomalies.reduce((acc, curr) => acc + curr.confidence, 0) / mockAnomalies.length).toFixed(2)),
};

// Generate mock explanations
export const mockExplanations: ExplanationData[] = mockAnomalies.map(anomaly => ({
  id: `explanation-${anomaly.id}`,
  anomalyId: anomaly.id,
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
}));

// Generate time series data for charts
export const generateTimeSeriesData = (days: number): TimeSeriesData[] => {
  return Array.from({ length: days }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (days - 1) + i);
    
    return {
      date: date.toISOString().split('T')[0],
      logs: Math.floor(Math.random() * 50) + 50,
      anomalies: Math.floor(Math.random() * 8) + 1,
    };
  });
};

export const timeSeriesData = generateTimeSeriesData(30);

// Generate real-time data stream simulation
export const generateRealtimeData = () => {
  const newLog: LogData = {
    id: `log-${Date.now()}`,
    timestamp: new Date().toISOString(),
    sourceIp: generateIp(),
    destinationIp: generateIp(),
    protocol: ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH'][Math.floor(Math.random() * 6)],
    encrypted: Math.random() > 0.3,
    size: Math.floor(Math.random() * 5000) + 500,
  };
  
  const isAnomaly = Math.random() < 0.15;
  
  const newAnomaly = isAnomaly ? {
    id: `anomaly-${Date.now()}`,
    timestamp: new Date().toISOString(),
    severity: ['critical', 'high', 'medium', 'low'][Math.floor(Math.random() * 4)] as AnomalyData['severity'],
    sourceIp: newLog.sourceIp,
    destinationIp: newLog.destinationIp,
    protocol: newLog.protocol,
    action: ['blocked', 'allowed', 'flagged', 'quarantined'][Math.floor(Math.random() * 4)],
    confidence: Math.round(Math.random() * 100) / 100,
    reviewed: false,
    details: [
      'Unusual traffic pattern detected between hosts',
      'Multiple failed login attempts',
      'Suspicious outbound data transfer',
      'Port scanning activity',
      'Potential data exfiltration',
      'Unusual API access pattern',
      'Credential stuffing attempt',
      'Brute force attack detected',
    ][Math.floor(Math.random() * 8)],
  } : null;
  
  return { newLog, newAnomaly };
};
