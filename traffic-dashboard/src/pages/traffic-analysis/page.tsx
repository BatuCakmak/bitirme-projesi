import { useState, useEffect, useMemo } from 'react';
import DashboardLayout from '@/components/feature/DashboardLayout';
import { useThemeColors } from '@/hooks/useThemeColors';

// Şablon Başlangıç Verileri (Yüklenirken hata vermemesi için)
const initialHourlyData = Array.from({ length: 24 }, (_, i) => ({
  hour: `${String(i).padStart(2, '0')}:00`, e5: 0, tem: 0, d100: 0, vatan: 0
}));

const initialRoadData = [
  { road: 'E-5 Karayolu', length: '42 km', density: 0, status: 'low' },
  { road: 'TEM Otoyolu', length: '38 km', density: 0, status: 'low' },
  { road: 'D-100', length: '56 km', density: 0, status: 'low' },
  { road: 'Vatan Caddesi', length: '8 km', density: 0, status: 'low' }
];

const lines = [
  { key: 'e5', label: 'E-5', color: '#00d4ff' },
  { key: 'tem', label: 'TEM', color: '#ff6b35' },
  { key: 'd100', label: 'D-100', color: '#00ff88' },
  { key: 'vatan', label: 'Vatan', color: '#ffd700' },
];

const W = 600, H = 180;

function toPath(data: number[], max: number) {
  if (max <= 0) max = 1; // Sıfıra bölünme hatasını önle
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - (v / max) * H;
    return `${x},${y}`;
  });
  return `M ${pts.join(' L ')}`;
}

export default function TrafficAnalysis() {
  const t = useThemeColors();

  // 1. Canlı Veri State'leri
  const [chartData, setChartData] = useState(initialHourlyData);
  const [roadData, setRoadData] = useState(initialRoadData);
  
  // Dinamik KPI Stat'leri
  const [kpiStats, setKpiStats] = useState({
    peakHour: 'Hesaplanıyor...',
    peakCount: 0,
    busiestRoad: 'Hesaplanıyor...',
    busiestPct: 0,
    fluentRoads: 0,
    totalRoads: 4
  });

  // 2. API'den Canlı Veri Çekme
  useEffect(() => {
    const fetchAnalysisData = async () => {
      try {
        // Saatlik Çizgi Grafik Verisini Çek
        const resHourly = await fetch('http://localhost:8080/api/vehicles/hourly');
        let fetchedHourly = initialHourlyData;
        if (resHourly.ok) {
          fetchedHourly = await resHourly.json();
          setChartData(fetchedHourly);
        }

        // Yol Yoğunluk Verisini Çek
        const resRoads = await fetch('http://localhost:8080/api/vehicles/road-density');
        let fetchedRoads = initialRoadData;
        if (resRoads.ok) {
          fetchedRoads = await resRoads.json();
          setRoadData(fetchedRoads);
        }

        // --- DİNAMİK KPI HESAPLAMA ---
        // 1. En çok aracın geçtiği saati (Pik Saat) bul
        let maxPeakCount = 0;
        let currentPeakHour = '00:00';
        fetchedHourly.forEach(h => {
          const totalInHour = (h.e5 || 0) + (h.tem || 0) + (h.d100 || 0) + (h.vatan || 0);
          if (totalInHour > maxPeakCount) {
            maxPeakCount = totalInHour;
            currentPeakHour = h.hour;
          }
        });

        // 2. En Yoğun Yolu ve Akıcı Yol Sayısını bul
        let maxDensity = -1;
        let busiestR = 'Veri Bekleniyor';
        let fluentCount = 0;

        fetchedRoads.forEach(r => {
          if (r.density > maxDensity) {
            maxDensity = r.density;
            busiestR = r.road;
          }
          if (r.status === 'low') fluentCount++;
        });

        // KPI State'i güncelle
        setKpiStats({
          peakHour: currentPeakHour,
          peakCount: maxPeakCount,
          busiestRoad: busiestR,
          busiestPct: maxDensity === -1 ? 0 : maxDensity,
          fluentRoads: fluentCount,
          totalRoads: fetchedRoads.length
        });

      } catch (error) {
        console.error("Analiz verileri çekilemedi:", error);
      }
    };

    fetchAnalysisData();
    const interval = setInterval(fetchAnalysisData, 3000); // 3 Saniyede bir yenile
    return () => clearInterval(interval);
  }, []);

  // Grafik yüksekliği için maksimum değeri hesapla
  const maxVal = Math.max(...chartData.map(d => Math.max(d.e5 || 0, d.tem || 0, d.d100 || 0, d.vatan || 0)));

  // Isı Haritası (Heatmap) Mock Verisi (Şimdilik sabit)
  const heatmapData = useMemo(() => {
    const days = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'];
    return days.map((day, di) => {
      const hours = Array.from({ length: 12 }, (_, hi) => {
        const intensity = Math.random();
        const isWeekend = di >= 5;
        const adj = isWeekend ? intensity * 0.5 : intensity;
        const color = adj > 0.7 ? '#ff4757' : adj > 0.4 ? '#ffd700' : '#00ff88';
        const opacity = 0.3 + adj * 0.7;
        return { color, opacity, title: `${day} ${hi * 2}:00` };
      });
      return { day, hours };
    });
  }, []);

  return (
    <DashboardLayout title="Trafik Analizi" subtitle="Trafik Yoğunluk Analizi ve Raporlar">
      <div className="space-y-4">
        
        {/* KPI row - ARTIK TAMAMEN DİNAMİK */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Günlük Pik', value: kpiStats.peakHour, sub: `${kpiStats.peakCount} araç/sa`, color: '#ff4757', icon: 'ri-time-line' },
            { label: 'En Yoğun Yol', value: kpiStats.busiestRoad, sub: `%${kpiStats.busiestPct} doluluk`, color: '#ffd700', icon: 'ri-road-map-line' },
            { label: 'Ort. Gecikme', value: '18 dk', sub: 'Normal: 8 dk', color: '#ff6b35', icon: 'ri-timer-line' }, // Bu Java'ya eklenebilir, şimdilik statik
            { label: 'Akıcı Yollar', value: `${kpiStats.fluentRoads} / ${kpiStats.totalRoads}`, sub: 'Canlı', color: '#00ff88', icon: 'ri-checkbox-circle-line' },
          ].map(k => (
            <div key={k.label} className={`${t.cardBg} rounded-xl p-4 border ${t.border} transition-colors duration-300`}>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 flex items-center justify-center rounded-lg" style={{ backgroundColor: `${k.color}20` }}>
                  <i className={`${k.icon} text-sm`} style={{ color: k.color }}></i>
                </div>
                <span className={`${t.textSecondary} text-xs`}>{k.label}</span>
              </div>
              <div className={`${t.textPrimary} font-bold text-base`}>{k.value}</div>
              <div className={`${t.textSecondary} text-xs mt-0.5`}>{k.sub}</div>
            </div>
          ))}
        </div>

        {/* Line chart */}
        <div className={`${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className={`${t.textPrimary} font-semibold text-sm`}>24 Saatlik Trafik Akışı</h3>
            <div className="flex gap-4">
              {lines.map(l => (
                <div key={l.key} className="flex items-center gap-1.5">
                  <div className="w-4 h-0.5 rounded" style={{ backgroundColor: l.color }}></div>
                  <span className={`${t.textSecondary} text-xs`}>{l.label}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto">
            <svg viewBox={`0 0 ${W} ${H + 20}`} className="w-full" style={{ minWidth: '500px' }}>
              {[0, 0.25, 0.5, 0.75, 1].map(pct => (
                <line key={pct} x1="0" y1={H * (1 - pct)} x2={W} y2={H * (1 - pct)} stroke={t.isDark ? '#1e2433' : '#e5e7eb'} strokeWidth="1" />
              ))}
              {lines.map(l => (
                <path
                  key={l.key}
                  d={toPath(chartData.map(d => (d as any)[l.key] || 0), maxVal)}
                  fill="none"
                  stroke={l.color}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  className="transition-all duration-500"
                />
              ))}
              {chartData.filter((_, i) => i % 3 === 0).map((d, i) => (
                <text key={i} x={(i * 3 / (chartData.length - 1)) * W} y={H + 16} fill={t.isDark ? '#8892a4' : '#6b7280'} fontSize="9" textAnchor="middle">{d.hour}</text>
              ))}
            </svg>
          </div>
        </div>

        {/* Road status */}
        <div className="grid grid-cols-2 gap-4">
          <div className={`${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm mb-4`}>Yol Durum Analizi</h3>
            <div className="space-y-4">
              {roadData.map(r => {
                const color = r.status === 'high' ? '#ff4757' : r.status === 'medium' ? '#ffd700' : '#00ff88';
                return (
                  <div key={r.road}>
                    <div className="flex justify-between mb-1">
                      <span className={`${t.textPrimary} text-xs font-medium`}>{r.road}</span>
                      <span className="text-xs font-bold transition-all duration-500" style={{ color }}>{r.density}%</span>
                    </div>
                    <div className={`${t.isDark ? 'bg-[#0f1117]' : 'bg-[#f3f4f6]'} rounded-full h-2`}>
                      <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${r.density}%`, backgroundColor: color }}></div>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className={`${t.textSecondary} text-[10px]`}>{r.length}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className={`${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm mb-4`}>Haftalık Yoğunluk Isı Haritası</h3>
            <div className="space-y-2">
              {heatmapData.map((dayData) => (
                <div key={dayData.day} className="flex items-center gap-2">
                  <span className={`${t.textSecondary} text-xs w-8`}>{dayData.day}</span>
                  <div className="flex gap-1 flex-1">
                    {dayData.hours.map((hourData, hi) => (
                      <div
                        key={hi}
                        className="flex-1 h-5 rounded-sm transition-opacity hover:opacity-100"
                        style={{ backgroundColor: hourData.color, opacity: hourData.opacity }}
                        title={hourData.title}
                      ></div>
                    ))}
                  </div>
                </div>
              ))}
              <div className="flex items-center gap-2 mt-2 justify-end">
                <span className={`${t.textSecondary} text-[10px]`}>Az</span>
                {['#00ff88','#ffd700','#ff4757'].map(c => <div key={c} className="w-4 h-2 rounded-sm" style={{ backgroundColor: c }}></div>)}
                <span className={`${t.textSecondary} text-[10px]`}>Çok</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}