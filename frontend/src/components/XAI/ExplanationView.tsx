
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from 'recharts';
import { AlertTriangle, BookOpen, Lightbulb, ShieldCheck } from 'lucide-react';
import { ExplanationData } from '../../data/mockData';

interface ExplanationViewProps {
  explanation: ExplanationData | null;
  isLoading?: boolean;
  modelSource?: 'mock' | 'trained'; // Add source indicator for future implementation
}

const ExplanationView: React.FC<ExplanationViewProps> = ({ 
  explanation, 
  isLoading = false,
  modelSource = 'mock' // Default to mock data for now
}) => {
  // Only process data if explanation exists
  const shapData = explanation?.shap?.map(item => ({
    name: item.feature.replace('_', ' '),
    value: parseFloat((item.importance * 100).toFixed(1))
  })) || [];

  const limeData = explanation?.lime?.map(item => ({
    name: item.feature.replace('_', ' '),
    value: parseFloat((item.importance * 100).toFixed(1))
  })) || [];

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="md:col-span-2">
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3 mb-2" />
            <div className="h-5 bg-muted/30 animate-pulse rounded-md w-1/2" />
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3" />
          </CardHeader>
          <CardContent>
            <div className="h-40 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3" />
          </CardHeader>
          <CardContent>
            <div className="h-40 bg-muted/30 animate-pulse rounded-md" />
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show a placeholder if explanation is null but not loading
  if (!explanation) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
              No Explanation Available
            </CardTitle>
            <CardDescription>
              Explanation data for this anomaly could not be loaded.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <AlertTriangle className="h-12 w-12 mb-4" />
              <p>Explanation data is missing or could not be processed.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center">
            <BookOpen className="mr-2 h-5 w-5 text-cyberpurple" />
            SHAP Feature Importance {modelSource === 'trained' && '(Trained Model)'}
          </CardTitle>
          <CardDescription>
            Showing how each feature contributes to the anomaly detection using SHAP (SHapley Additive exPlanations)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={shapData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis 
                  type="number" 
                  unit="%" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, 'Importance']}
                  contentStyle={{ 
                    backgroundColor: 'var(--card)', 
                    borderColor: 'var(--border)',
                    color: 'var(--card-foreground)'
                  }}
                  labelStyle={{ color: 'var(--card-foreground)' }}
                />
                <Bar dataKey="value" fill="#8B5CF6">
                  <LabelList dataKey="value" position="right" formatter={(value: number) => `${value}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
            Contributing Factors {modelSource === 'trained' && '(Trained Model)'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {explanation.contributingFactors.map((factor, index) => (
              <li key={index} className="flex items-start">
                <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-amber-500/10 text-amber-500 mr-2">
                  {index + 1}
                </span>
                <span>{factor}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <ShieldCheck className="mr-2 h-5 w-5 text-green-500" />
            Recommendations {modelSource === 'trained' && '(Trained Model)'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {explanation.recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start">
                <Lightbulb className="h-5 w-5 text-green-500 mr-2 shrink-0" />
                <span>{recommendation}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
      
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center">
            <BookOpen className="mr-2 h-5 w-5 text-blue-500" />
            LIME Feature Importance {modelSource === 'trained' && '(Trained Model)'}
          </CardTitle>
          <CardDescription>
            Showing how each feature contributes to the anomaly detection using LIME (Local Interpretable Model-agnostic Explanations)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={limeData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis 
                  type="number" 
                  unit="%" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  tick={{ fill: 'var(--muted-foreground)' }}
                  axisLine={{ stroke: 'var(--border)' }}
                  tickLine={{ stroke: 'var(--border)' }}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, 'Importance']}
                  contentStyle={{ 
                    backgroundColor: 'var(--card)', 
                    borderColor: 'var(--border)',
                    color: 'var(--card-foreground)'
                  }}
                  labelStyle={{ color: 'var(--card-foreground)' }}
                />
                <Bar dataKey="value" fill="#3B82F6">
                  <LabelList dataKey="value" position="right" formatter={(value: number) => `${value}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ExplanationView;
