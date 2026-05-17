import DashboardLayout from '@/components/feature/DashboardLayout';
import CameraGrid from './components/CameraGrid';
import StatsBar from './components/StatsBar';
import LiveAlerts from './components/LiveAlerts';
import TrafficMap from './components/TrafficMap';

export default function Home() {
  return (
    <DashboardLayout title="Canlı Harita" subtitle="Canlı Kamera İzleme ve Trafik Haritası">
      <div className="space-y-4">
        <StatsBar />
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 space-y-4">
            <CameraGrid />
            <TrafficMap />
          </div>
          <div>
            <LiveAlerts />
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
