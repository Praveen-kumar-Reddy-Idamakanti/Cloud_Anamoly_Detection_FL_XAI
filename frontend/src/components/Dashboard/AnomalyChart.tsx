
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';
import { TimeSeriesData } from '../../data/mockData';

interface AnomalyChartProps {
  data: TimeSeriesData[];
  isLoading?: boolean;
  title: string;
  type?: 'area' | 'bar';
}

const AnomalyChart: React.FC<AnomalyChartProps> = ({
  data,
  isLoading = false,
  title,
  type = 'area',
}) => {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <div className="w-full h-full bg-muted/30 animate-pulse rounded-md" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'area' ? (
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorLogs" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorAnomalies" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff4557" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#ff4557" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="date" 
                tick={{ fill: 'var(--muted-foreground)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
              />
              <YAxis 
                tick={{ fill: 'var(--muted-foreground)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
              />
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--card)', 
                  borderColor: 'var(--border)',
                  color: 'var(--card-foreground)'
                }}
                labelStyle={{ color: 'var(--card-foreground)' }}
              />
              <Area 
                type="monotone" 
                dataKey="logs" 
                stroke="#8884d8" 
                fillOpacity={1} 
                fill="url(#colorLogs)" 
                name="Logs"
              />
              <Area 
                type="monotone" 
                dataKey="anomalies" 
                stroke="#ff4557" 
                fillOpacity={1} 
                fill="url(#colorAnomalies)" 
                name="Anomalies"
              />
            </AreaChart>
          ) : (
            <BarChart data={data}>
              <XAxis 
                dataKey="date" 
                tick={{ fill: 'var(--muted-foreground)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
              />
              <YAxis 
                tick={{ fill: 'var(--muted-foreground)' }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
              />
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--card)', 
                  borderColor: 'var(--border)',
                  color: 'var(--card-foreground)'
                }}
                labelStyle={{ color: 'var(--card-foreground)' }}
              />
              <Legend />
              <Bar dataKey="logs" fill="#8884d8" name="Logs" />
              <Bar dataKey="anomalies" fill="#ff4557" name="Anomalies" />
            </BarChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default AnomalyChart;
