
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, BookOpen, CheckCircle2, Clock } from 'lucide-react';
import { toast } from 'sonner';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import ExplanationView from '../components/XAI/ExplanationView';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { anomaliesApi, explanationsApi } from '../api/api';
import { AnomalyData, ExplanationData } from '../data/mockData';

const XAIExplanation: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [anomaly, setAnomaly] = useState<AnomalyData | null>(null);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!id) return;
      
      setIsLoading(true);
      try {
        // Fetch anomaly details
        const anomalyData = await anomaliesApi.getAnomalyById(id);
        setAnomaly(anomalyData);
        
        // Fetch explanation
        try {
          const explanationData = await explanationsApi.getExplanation(id);
          setExplanation(explanationData);
        } catch (error: any) {
          console.error("Failed to fetch explanation:", error);
          // Still set explanation to null but don't show error toast
          // This allows the UI to handle missing explanations gracefully
          setExplanation(null);
        }
      } catch (error: any) {
        toast.error(error.message || 'Failed to fetch anomaly data');
        setAnomaly(null);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [id]);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const getSeverityColor = (severity?: AnomalyData['severity']) => {
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

  const formatDate = (dateString?: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const handleMarkAsReviewed = async () => {
    if (!anomaly) return;
    
    try {
      const updatedAnomaly = await anomaliesApi.reviewAnomaly(anomaly.id, true);
      setAnomaly(updatedAnomaly);
      toast.success('Anomaly marked as reviewed');
    } catch (error: any) {
      toast.error(error.message || 'Failed to update anomaly status');
    }
  };

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="mb-6">
            <Button asChild variant="outline" size="sm" className="mb-4">
              <Link to="/anomalies">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Anomalies
              </Link>
            </Button>
            
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center">
                <BookOpen className="h-6 w-6 mr-2 text-cyberpurple" />
                <h1 className="text-2xl font-bold">XAI Explanation</h1>
              </div>
              
              {anomaly && !anomaly.reviewed && (
                <Button onClick={handleMarkAsReviewed}>
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  Mark as Reviewed
                </Button>
              )}
            </div>
          </div>

          {isLoading ? (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="h-7 bg-muted/30 animate-pulse rounded-md w-1/3 mb-2" />
                  <div className="h-5 bg-muted/30 animate-pulse rounded-md w-1/2" />
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <div key={i} className="h-20 bg-muted/30 animate-pulse rounded-md" />
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <ExplanationView explanation={null} isLoading={true} />
            </div>
          ) : anomaly ? (
            <div className="space-y-6">
              {/* Anomaly Details */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
                    Anomaly Details
                  </CardTitle>
                  <CardDescription>
                    Detection information and network details for this security event
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Severity</p>
                      <Badge className={getSeverityColor(anomaly.severity)} variant="outline">
                        {anomaly.severity.charAt(0).toUpperCase() + anomaly.severity.slice(1)}
                      </Badge>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Status</p>
                      <div className="flex items-center">
                        {anomaly.reviewed ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500 mr-1" />
                        ) : (
                          <Clock className="h-4 w-4 text-amber-500 mr-1" />
                        )}
                        <span>{anomaly.reviewed ? 'Reviewed' : 'Pending Review'}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <div>
                        <div className="w-full bg-secondary rounded-full h-2.5">
                          <div
                            className="bg-cyberpurple h-2.5 rounded-full"
                            style={{ width: `${anomaly.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span>{(anomaly.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Timestamp</p>
                      <p className="font-mono text-sm">{formatDate(anomaly.timestamp)}</p>
                    </div>
                  </div>
                  
                  <Separator className="my-4" />
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Network Information</h3>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-sm text-muted-foreground">Source IP</p>
                          <p className="font-mono text-sm">{anomaly.sourceIp}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Destination IP</p>
                          <p className="font-mono text-sm">{anomaly.destinationIp}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Protocol</p>
                          <p className="text-sm">{anomaly.protocol}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Action</p>
                          <p className="text-sm">{anomaly.action}</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <h3 className="text-sm font-medium">Details</h3>
                      <p>{anomaly.details}</p>
                      
                      {explanation && (
                        <div>
                          <p className="text-sm text-muted-foreground">Model Type</p>
                          <p className="text-sm">{explanation.modelType}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              {/* XAI Explanation */}
              <ExplanationView explanation={explanation} />
            </div>
          ) : (
            <div className="text-center py-12">
              <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-xl font-medium mb-2">Anomaly Not Found</h2>
              <p className="text-muted-foreground mb-6">
                We couldn't find this anomaly in our system.
              </p>
              <Button asChild variant="outline">
                <Link to="/anomalies">Back to Anomalies</Link>
              </Button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default XAIExplanation;
