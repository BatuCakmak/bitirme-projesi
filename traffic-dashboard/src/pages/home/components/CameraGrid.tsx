import { useState, useEffect } from 'react';
import { cameras } from '@/mocks/cameras';
import { useThemeColors } from '@/hooks/useThemeColors';

const densityConfig = {
  high: { label: 'Yoğun', color: 'text-[#ff4757]', bg: 'bg-[#ff4757]/20', dot: 'bg-[#ff4757]' },
  medium: { label: 'Orta', color: 'text-[#ffd700]', bg: 'bg-[#ffd700]/20', dot: 'bg-[#ffd700]' },
  low: { label: 'Akıcı', color: 'text-[#00ff88]', bg: 'bg-[#00ff88]/20', dot: 'bg-[#00ff88]' },
};

const statusConfig = {
  active: { label: 'CANLI', color: 'text-[#ff4757]', dot: 'bg-[#ff4757]' },
  warning: { label: 'UYARI', color: 'text-[#ffd700]', dot: 'bg-[#ffd700]' },
  offline: { label: 'KAPALI', color: 'text-[#8892a4]', dot: 'bg-[#8892a4]' },
};

export default function CameraGrid() {
  const [fullscreen, setFullscreen] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const t = useThemeColors();

  // YENİ: Kameraya özel verileri (Akış Hızı ve Toplam Araç) tutan state
  const [liveData, setLiveData] = useState<{
    densities: Record<string, { status: string, pct: number, totalCount: number, flowRate: number }>
  }>({
    densities: {}
  });

  useEffect(() => {
    const fetchCameraData = async () => {
      try {
        const densityRes = await fetch('http://localhost:8080/api/vehicles/road-density');
        const densityMap: Record<string, { status: string, pct: number, totalCount: number, flowRate: number }> = {};
        
        if (densityRes.ok) {
          const data = await densityRes.json();
          data.forEach((item: any) => {
            // Artık API'den cameraId geldiği için eşleştirme çok daha güvenli
            if (item.cameraId) {
              densityMap[item.cameraId] = { 
                status: item.status, 
                pct: item.density,
                totalCount: item.totalCount, // O kameranın toplam aracı
                flowRate: item.flowRate      // O kameranın akış hızı
              };
            }
          });
        }

        setLiveData({ densities: densityMap });
      } catch (error) {
        console.error("Kamera veri çekme hatası:", error);
      }
    };

    fetchCameraData();
    const interval = setInterval(fetchCameraData, 3000);
    return () => clearInterval(interval);
  }, []);

  const displayCameras = cameras.slice(0, 6);

  return (
    <>
      <div className="grid grid-cols-3 gap-3">
        {displayCameras.map((cam) => {
          const camLiveData = liveData.densities[cam.id];
          const isLiveCam = cam.id === 'CAM-001' || cam.id === 'CAM-002';
          
          // YENİ: Kameranın kendi özel araç sayısı çekiliyor
          const currentVehicleCount = (isLiveCam && camLiveData) ? camLiveData.totalCount : cam.vehicleCount;
          const currentDensityKey = camLiveData ? camLiveData.status : cam.density;

          const density = densityConfig[currentDensityKey as keyof typeof densityConfig];
          const status = statusConfig[cam.status as keyof typeof statusConfig];
          
          const imageSource = (cam as any).streamUrl || cam.thumbnail;

          return (
            <div
              key={cam.id}
              className={`relative rounded-xl overflow-hidden cursor-pointer group transition-all duration-200 border ${
                selected === cam.id ? 'border-[#00d4ff]' : `${t.border} hover:border-[#00d4ff]/50`
              }`}
              onClick={() => setSelected(selected === cam.id ? null : cam.id)}
            >
              <div className="relative w-full h-0 pb-[56.25%]">
                <img
                  src={imageSource}
                  alt={cam.name}
                  className="absolute inset-0 w-full h-full object-cover object-top"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = cam.thumbnail;
                  }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30"></div>

                <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-2">
                  <div className="bg-black/60 backdrop-blur-sm rounded px-2 py-1">
                    <div className="text-white text-[10px] font-bold">{cam.id}</div>
                    <div className="text-[#ccc] text-[9px]">{cam.location}</div>
                  </div>
                  <div className="flex items-center gap-1 bg-black/60 backdrop-blur-sm rounded px-2 py-1">
                    <div className={`w-1.5 h-1.5 rounded-full ${status.dot} animate-pulse`}></div>
                    <span className={`text-[9px] font-bold ${status.color}`}>{status.label}</span>
                  </div>
                </div>

                <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-3 py-2">
                  <div>
                    <div className="text-white text-xs font-semibold">{cam.name}</div>
                    {/* YENİ: Hem Toplam Sayı Hem de Akış Hızı Ayrı Ayrı Gösteriliyor */}
                    <div className="text-[#ccc] text-[10px] mt-0.5 flex items-center gap-2">
                      <span>{currentVehicleCount} araç</span>
                      {isLiveCam && camLiveData && (
                        <span className="text-[#00d4ff] font-medium">• {camLiveData.flowRate} araç/dk</span>
                      )}
                    </div>
                  </div>
                  <div className={`flex items-center gap-1 rounded-full px-2 py-0.5 ${density.bg}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${density.dot}`}></div>
                    <span className={`text-[10px] font-semibold ${density.color}`}>
                      {density.label} {camLiveData?.pct !== undefined ? `(%${camLiveData.pct})` : ''}
                    </span>
                  </div>
                </div>

                <button
                  className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-white text-xs flex items-center gap-2"
                  onClick={(e) => { e.stopPropagation(); setFullscreen(cam.id); }}
                >
                  <i className="ri-fullscreen-line"></i>
                  Tam Ekran
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* TAM EKRAN MODU */}
      {fullscreen && (() => {
        const cam = cameras.find(c => c.id === fullscreen);
        if (!cam) return null;

        const camLiveData = liveData.densities[cam.id];
        const isLiveCam = cam.id === 'CAM-001' || cam.id === 'CAM-002';
        
        const currentVehicleCount = (isLiveCam && camLiveData) ? camLiveData.totalCount : cam.vehicleCount;
        const currentDensityKey = camLiveData ? camLiveData.status : cam.density;

        const density = densityConfig[currentDensityKey as keyof typeof densityConfig];
        const imageSource = (cam as any).streamUrl || cam.thumbnail;

        return (
          <div
            className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8"
            onClick={() => setFullscreen(null)}
          >
            <div className="relative w-full max-w-5xl rounded-2xl overflow-hidden border border-[#1e2433]" onClick={e => e.stopPropagation()}>
              <img 
                src={imageSource} 
                alt={cam.name} 
                className="w-full object-cover object-top"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = cam.thumbnail;
                }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30"></div>
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
                <div>
                  <div className="text-white font-bold text-lg">{cam.name}</div>
                  <div className="text-[#ccc] text-sm">{cam.location}</div>
                </div>
                <button onClick={() => setFullscreen(null)} className="w-10 h-10 flex items-center justify-center bg-black/60 rounded-lg text-white hover:bg-white/20 transition-colors cursor-pointer">
                  <i className="ri-close-line text-xl"></i>
                </button>
              </div>
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="bg-black/60 rounded-lg px-3 py-2 text-center">
                    <div className="text-[#ccc] text-xs">Geçen Araç</div>
                    <div className="text-white font-bold text-xl">{currentVehicleCount}</div>
                  </div>
                  {/* Tam Ekrana da Akış Hızını Ekledik */}
                  {isLiveCam && camLiveData && (
                    <div className="bg-black/60 rounded-lg px-3 py-2 text-center">
                      <div className="text-[#ccc] text-xs">Akış Hızı</div>
                      <div className="text-[#00d4ff] font-bold text-xl">{camLiveData.flowRate} <span className="text-sm font-normal text-[#ccc]">dk</span></div>
                    </div>
                  )}
                </div>
                <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 ${density.bg}`}>
                  <div className={`w-2 h-2 rounded-full ${density.dot}`}></div>
                  <span className={`text-sm font-semibold ${density.color}`}>
                    {density.label} {camLiveData?.pct !== undefined ? `(%${camLiveData.pct})` : ''}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </>
  );
}