
import React from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, CheckCircle2, ChevronRight, Clock } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AnomalyData } from '../../data/mockData';

interface AnomalyTableProps {
  anomalies: AnomalyData[];
  isLoading?: boolean;
}

const AnomalyTable: React.FC<AnomalyTableProps> = ({ anomalies, isLoading = false }) => {
  const getSeverityColor = (severity: AnomalyData['severity']) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'high':
        return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20';
      case 'low':
        return 'bg-green-500/10 text-green-500 border-green-500/20';
      default:
        return 'bg-secondary text-secondary-foreground';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  if (isLoading) {
    return (
      <div className="space-y-3">
        <div className="h-10 bg-muted/30 animate-pulse rounded-md w-1/3" />
        <div className="border rounded-md">
          <div className="h-12 border-b bg-muted/30 animate-pulse" />
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 border-b bg-muted/30 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-medium mb-4">Recent Anomalies</h3>
      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Severity</TableHead>
              <TableHead>Timestamp</TableHead>
              <TableHead>Source IP</TableHead>
              <TableHead>Protocol</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {anomalies.slice(0, 5).map((anomaly) => (
              <TableRow key={anomaly.id}>
                <TableCell>
                  <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
                    {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono text-xs">{formatDate(anomaly.timestamp)}</TableCell>
                <TableCell className="font-mono text-xs">{anomaly.sourceIp}</TableCell>
                <TableCell>{anomaly.protocol}</TableCell>
                <TableCell>
                  <div className="flex items-center">
                    {anomaly.reviewed ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                    ) : (
                      <Clock className="h-4 w-4 text-amber-500 mr-1" />
                    )}
                    <span className="text-sm">{anomaly.reviewed ? 'Reviewed' : 'Pending'}</span>
                  </div>
                </TableCell>
                <TableCell>
                  <div className="w-full bg-secondary rounded-full h-2.5">
                    <div
                      className="bg-cyberpurple h-2.5 rounded-full"
                      style={{ width: `${anomaly.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-muted-foreground">{(anomaly.confidence * 100).toFixed(0)}%</span>
                </TableCell>
                <TableCell className="text-right">
                  <Button asChild variant="ghost" size="sm">
                    <Link to={`/explanations/${anomaly.id}`}>
                      <span className="mr-1">Details</span>
                      <ChevronRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <div className="mt-4 text-center">
        <Button asChild variant="outline">
          <Link to="/anomalies">View All Anomalies</Link>
        </Button>
      </div>
    </div>
  );
};

export default AnomalyTable;
