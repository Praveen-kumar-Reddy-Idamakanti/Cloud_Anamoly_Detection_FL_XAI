
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, 
  ShieldAlert, 
  AreaChart, 
  LineChart, 
  UploadCloud,
  AlertTriangle,
  Network
} from 'lucide-react';
import { toast } from 'sonner';
import { useAuth } from '../contexts/AuthContext';
import Navbar from '../components/Layout/Navbar';
import Sidebar from '../components/Layout/Sidebar';
import StatCard from '../components/Dashboard/StatCard';
import AnomalyChart from '../components/Dashboard/AnomalyChart';
import LogUpload from '../components/Dashboard/LogUpload';
import AnomalyTable from '../components/Dashboard/AnomalyTable';
import { statsApi, anomaliesApi, realtimeApi } from '../api/api';
import { StatData, AnomalyData, TimeSeriesData } from '../data/mockData';

const Dashboard: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<StatData | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyData[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [newAnomalyAlert, setNewAnomalyAlert] = useState<AnomalyData | null>(null);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!isAuthenticated) {
      navigate('/login');
      return;
    }

    // Fetch initial dashboard data
    const fetchDashboardData = async () => {
      setIsLoading(true);
      try {
        // Fetch stats
        const statsData = await statsApi.getStats();
        setStats(statsData);

        // Fetch anomalies
        const anomaliesData = await anomaliesApi.getAnomalies(1, 5);
        setAnomalies(anomaliesData.data);

        // Fetch time series data
        const timeData = await statsApi.getTimeSeriesData(30);
        setTimeSeriesData(timeData);

      } catch (error: any) {
        toast.error(error.message || 'Failed to load dashboard data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();

    // Subscribe to real-time updates
    const unsubscribe = realtimeApi.subscribeToRealTimeUpdates(({ log, anomaly }) => {
      if (anomaly) {
        // Update anomalies list
        setAnomalies(prev => [anomaly, ...prev.slice(0, 4)]);
        
        // Show notification
        toast('New anomaly detected', {
          description: `${anomaly.severity.toUpperCase()} severity from ${anomaly.sourceIp}`,
          action: {
            label: 'View',
            onClick: () => navigate(`/explanations/${anomaly.id}`),
          },
        });
        
        // Set the alert
        setNewAnomalyAlert(anomaly);
        setTimeout(() => setNewAnomalyAlert(null), 5000);
      }
    });

    return () => {
      // Clean up real-time subscription
      unsubscribe();
    };
  }, [isAuthenticated, navigate]);

  // Toggle sidebar
  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="min-h-screen">
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar isSidebarOpen={isSidebarOpen} />

      <main className={`pt-16 transition-all duration-300 ${isSidebarOpen ? 'md:ml-64' : 'md:ml-16'}`}>
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
          <div className="flex items-center mb-6">
            <LayoutDashboard className="h-6 w-6 mr-2 text-cyberpurple" />
            <h1 className="text-2xl font-bold">Dashboard</h1>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard
              title="Total Logs"
              value={stats?.totalLogs || 0}
              icon={<Network className="h-4 w-4" />}
              isLoading={isLoading}
              change={2.5}
            />
            <StatCard
              title="Total Anomalies"
              value={stats?.totalAnomalies || 0}
              icon={<AlertTriangle className="h-4 w-4" />}
              isLoading={isLoading}
              variant="danger"
              change={-1.2}
            />
            <StatCard
              title="Alert Rate"
              value={`${stats?.alertRate || 0}%`}
              icon={<ShieldAlert className="h-4 w-4" />}
              isLoading={isLoading}
              variant="warning"
              change={0.8}
            />
            <StatCard
              title="Avg. Confidence"
              value={`${stats?.avgConfidence || 0}%`}
              icon={<AreaChart className="h-4 w-4" />}
              isLoading={isLoading}
              variant="primary"
              change={3.4}
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <div className="lg:col-span-2">
              <AnomalyChart
                title="Network Activity"
                data={timeSeriesData}
                isLoading={isLoading}
              />
            </div>
            <div>
              <LogUpload />
            </div>
          </div>

          {/* Anomalies Table */}
          <div className="mb-6">
            <AnomalyTable anomalies={anomalies} isLoading={isLoading} />
          </div>
        </div>
      </main>

      {/* New Anomaly Alert */}
      {newAnomalyAlert && (
        <div className="fixed bottom-4 right-4 max-w-sm w-full bg-card border border-red-500 rounded-lg p-4 shadow-lg animate-fade-in">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <AlertTriangle className="h-6 w-6 text-red-500" />
            </div>
            <div className="ml-3 w-0 flex-1">
              <p className="text-sm font-medium">New Anomaly Detected</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {newAnomalyAlert.severity.toUpperCase()} severity from {newAnomalyAlert.sourceIp}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
