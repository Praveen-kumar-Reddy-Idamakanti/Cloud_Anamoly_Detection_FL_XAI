
import React, { useState, useEffect } from 'react';
import { Database, Play, Pause, MoreHorizontal, PlusCircle, RefreshCw, BarChart, Info } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useAuth } from '../contexts/AuthContext';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Mock model data
const mockModels = [
  { 
    id: '1', 
    name: 'Isolation Forest', 
    type: 'anomaly', 
    version: '1.0.3',
    status: 'running',
    accuracy: 87.5,
    lastTrained: '2024-03-15T15:30:00',
    dataPoints: 15843,
    cpu: 15,
    memory: 28
  },
  { 
    id: '2', 
    name: 'Local Outlier Factor', 
    type: 'anomaly', 
    version: '2.1.0',
    status: 'running',
    accuracy: 91.2,
    lastTrained: '2024-04-01T09:45:00',
    dataPoints: 22150,
    cpu: 24,
    memory: 42
  },
  { 
    id: '3', 
    name: 'One-Class SVM', 
    type: 'anomaly', 
    version: '1.2.5',
    status: 'stopped',
    accuracy: 85.7,
    lastTrained: '2024-02-20T11:15:00',
    dataPoints: 9876,
    cpu: 0,
    memory: 0
  },
  { 
    id: '4', 
    name: 'Neural Network', 
    type: 'classification', 
    version: '3.0.1',
    status: 'training',
    accuracy: 89.3,
    lastTrained: '2024-04-09T14:20:00',
    dataPoints: 31245,
    cpu: 75,
    memory: 85
  },
];

const ModelManagement: React.FC = () => {
  const { isAuthenticated, user } = useAuth();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [models, setModels] = useState(mockModels);
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    // Check if user is admin
    if (user?.role !== 'admin') {
      navigate('/dashboard');
      toast.error('You do not have access to this page');
      return;
    }
  }, [isAuthenticated, navigate, user]);

  useEffect(() => {
    if (isTrainingModel) {
      const interval = setInterval(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + 10;
          if (newProgress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
              setIsTrainingModel(false);
              toast.success('Model training completed');
              // Update the model that was training
              setModels(prevModels => 
                prevModels.map(model => 
                  model.status === 'training' 
                    ? { ...model, status: 'running', lastTrained: new Date().toISOString(), accuracy: Math.min(model.accuracy + 2.5, 99.9) } 
                    : model
                )
              );
            }, 500);
            return 100;
          }
          return newProgress;
        });
      }, 800);

      return () => clearInterval(interval);
    }
  }, [isTrainingModel]);

  // Toggle sidebar
  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  // Toggle model status
  const toggleModelStatus = (id: string) => {
    const updatedModels = models.map(model => {
      if (model.id === id) {
        const newStatus = model.status === 'running' ? 'stopped' : 'running';
        return { 
          ...model, 
          status: newStatus,
          cpu: newStatus === 'running' ? Math.floor(Math.random() * 30) + 10 : 0,
          memory: newStatus === 'running' ? Math.floor(Math.random() * 40) + 20 : 0
        };
      }
      return model;
    });
    
    setModels(updatedModels);
    const targetModel = models.find(m => m.id === id);
    const statusText = targetModel?.status === 'running' ? 'stopped' : 'started';
    toast.success(`Model ${targetModel?.name} ${statusText} successfully`);
  };

  // Start model training
  const startTraining = (id: string) => {
    const updatedModels = models.map(model => {
      if (model.id === id) {
        return { ...model, status: 'training' };
      }
      return model;
    });
    
    setModels(updatedModels);
    setIsTrainingModel(true);
    setTrainingProgress(0);
    toast.info('Model training started', {
      description: 'This process may take several minutes to complete'
    });
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    }).format(date);
  };

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Database className="h-6 w-6 mr-2 text-cyberpurple" />
              <h1 className="text-2xl font-bold">Model Management</h1>
            </div>
            
            <Button onClick={() => toast.info('New model deployment wizard would open here')}>
              <PlusCircle className="h-4 w-4 mr-2" />
              Deploy New Model
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Active Models</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {models.filter(m => m.status === 'running').length} / {models.length}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Models currently processing data
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg. Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(models.reduce((acc, m) => acc + m.accuracy, 0) / models.length).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Across all deployed models
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Total Data Points</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {models.reduce((acc, m) => acc + m.dataPoints, 0).toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Used for model training
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Resource Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span>CPU</span>
                      <span>{Math.round(models.reduce((acc, m) => acc + m.cpu, 0) / models.length)}%</span>
                    </div>
                    <Progress value={models.reduce((acc, m) => acc + m.cpu, 0) / models.length} className="h-1" />
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span>Memory</span>
                      <span>{Math.round(models.reduce((acc, m) => acc + m.memory, 0) / models.length)}%</span>
                    </div>
                    <Progress value={models.reduce((acc, m) => acc + m.memory, 0) / models.length} className="h-1" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="models" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
            </TabsList>
            
            <TabsContent value="models">
              <Card>
                <CardHeader>
                  <CardTitle>Deployed Models</CardTitle>
                  <CardDescription>
                    Manage and monitor your deployed AI models.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Version</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Accuracy</TableHead>
                        <TableHead>Last Trained</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {models.map((model) => (
                        <TableRow key={model.id}>
                          <TableCell className="font-medium">{model.name}</TableCell>
                          <TableCell>{model.type}</TableCell>
                          <TableCell>v{model.version}</TableCell>
                          <TableCell>
                            {model.status === 'running' && (
                              <Badge variant="outline" className="bg-green-100 text-green-800 hover:bg-green-100">
                                <span className="flex items-center">
                                  <span className="relative flex h-2 w-2 mr-1">
                                    <span className="absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75 animate-ping"></span>
                                    <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500"></span>
                                  </span>
                                  Running
                                </span>
                              </Badge>
                            )}
                            {model.status === 'stopped' && (
                              <Badge variant="outline" className="bg-slate-100 text-slate-800 hover:bg-slate-100">
                                Stopped
                              </Badge>
                            )}
                            {model.status === 'training' && (
                              <Badge variant="outline" className="bg-blue-100 text-blue-800 hover:bg-blue-100">
                                <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                                Training
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center">
                              <span className="mr-2">{model.accuracy.toFixed(1)}%</span>
                              <Progress value={model.accuracy} className="h-1 w-16" />
                            </div>
                          </TableCell>
                          <TableCell>{formatDate(model.lastTrained)}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              {model.status !== 'training' && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => toggleModelStatus(model.id)}
                                  title={model.status === 'running' ? 'Stop model' : 'Start model'}
                                >
                                  {model.status === 'running' ? (
                                    <Pause className="h-4 w-4" />
                                  ) : (
                                    <Play className="h-4 w-4" />
                                  )}
                                </Button>
                              )}
                              
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => startTraining(model.id)}
                                disabled={model.status === 'training' || isTrainingModel}
                                title="Retrain model"
                              >
                                <RefreshCw className="h-4 w-4" />
                              </Button>
                              
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => toast.info(`View ${model.name} details`)}
                                title="View model details"
                              >
                                <Info className="h-4 w-4" />
                              </Button>
                              
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem onClick={() => toast.info(`View ${model.name} performance`)}>
                                    <BarChart className="h-4 w-4 mr-2" />
                                    Performance Metrics
                                  </DropdownMenuItem>
                                  <DropdownMenuItem onClick={() => toast.info(`Downloading model config for ${model.name}`)}>
                                    Export Configuration
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem onClick={() => toast.info(`Updating ${model.name}`)}>
                                    Update Version
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
                
                {isTrainingModel && (
                  <CardFooter className="border-t p-4">
                    <div className="w-full space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="flex items-center">
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Training model...
                        </span>
                        <span>{trainingProgress}%</span>
                      </div>
                      <Progress value={trainingProgress} className="h-2" />
                    </div>
                  </CardFooter>
                )}
              </Card>
            </TabsContent>
            
            <TabsContent value="performance">
              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                  <CardDescription>
                    Detailed performance metrics for each model.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-10 text-muted-foreground">
                    <BarChart className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
                    <h3 className="text-lg font-medium mb-2">Performance Metrics</h3>
                    <p>Detailed analytics charts would appear here, showing accuracy, F1 score, precision/recall, and other metrics.</p>
                    <Button 
                      variant="outline" 
                      className="mt-4"
                      onClick={() => toast.info('Performance dashboard would open here')}
                    >
                      Open Full Analytics
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
};

export default ModelManagement;
