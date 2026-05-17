import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/feature/DashboardLayout';
import { hourlyTrafficData, dailyVehicleCount, roadDensity } from '@/mocks/traffic';
import { useThemeColors } from '@/hooks/useThemeColors';

const densityConfig = {
  high: { label: 'Yoğun', color: 'text-[#ff4757]', bg: 'bg-[#ff4757]/20', bar: 'bg-[#ff4757]' },
  medium: { label: 'Orta', color: 'text-[#ffd700]', bg: 'bg-[#ffd700]/20', bar: 'bg-[#ffd700]' },
  low: { label: 'Akıcı', color: 'text-[#00ff88]', bg: 'bg-[#00ff88]/20', bar: 'bg-[#00ff88]' },
};

export default function VehicleCount() {
  const [activeRoad, setActiveRoad] = useState('e5');
  const t = useThemeColors();

  // 1. Verileri State İçine Alıyoruz
  const [kpiStats, setKpiStats] = useState({
    todayTotal: 13595,
    thisWeek: '84,350',
    thisMonth: '342,180',
    peakHour: '08:00'
  });
  const [hourlyData, setHourlyData] = useState(hourlyTrafficData);
  const [dailyData, setDailyData] = useState(dailyVehicleCount);
  const [roadData, setRoadData] = useState(roadDensity);

  // Maksimum değerleri state içindeki verilere göre dinamik hesapla
  const maxHourly = Math.max(...hourlyData.map(d => Math.max(d.e5, d.tem, d.d100, d.vatan)));
  const maxDaily = Math.max(...dailyData.map(d => d.count));

  const roadKeys = [
    { key: 'e5', label: 'E-5', color: '#00d4ff' },
    { key: 'tem', label: 'TEM', color: '#ff6b35' },
    { key: 'd100', label: 'D-100', color: '#00ff88' },
    { key: 'vatan', label: 'Vatan', color: '#ffd700' },
  ];

  // 2. API'den Canlı Veri Çekme (Polling)
  useEffect(() => {
    const fetchCounts = async () => {
      try {
        // Mevcut Spring Boot API'mizden toplam araç sayısını çekiyoruz (Bugün Toplam)
        const res = await fetch('http://localhost:8080/api/vehicles/stats');
        if (res.ok) {
          const data = await res.json();
          setKpiStats(prev => ({
            ...prev,
            todayTotal: data.toplamArac || 0
          }));
        }

        // Saatlik Araç Sayımı Grafiği İçin
        const resHourly = await fetch('http://localhost:8080/api/vehicles/hourly');
        if(resHourly.ok) setHourlyData(await resHourly.json());
        
        // Günlük Karşılaştırma Grafiği İçin
        const resDaily = await fetch('http://localhost:8080/api/vehicles/daily-comparison');
        if(resDaily.ok) setDailyData(await resDaily.json());

        // --- YENİ EKLENEN KISIM: Yol Bazlı Araç Yoğunluğu İçin ---
        const resRoads = await fetch('http://localhost:8080/api/vehicles/road-density');
        if(resRoads.ok) {
           const roadJson = await resRoads.json();
           setRoadData(roadJson);
        }

      } catch (error) {
        console.error("Araç sayım verileri çekilemedi:", error);
      }
    };

    fetchCounts();
    const interval = setInterval(fetchCounts, 3000); 
    return () => clearInterval(interval);
  }, []);

  return (
    <DashboardLayout title="Araç Sayımı" subtitle="Araç Sayım İstatistikleri">
      <div className="space-y-4">
        
        {/* KPI Row (Bugün Toplam artık dinamik) */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Bugün Toplam', value: kpiStats.todayTotal.toLocaleString('tr-TR'), icon: 'ri-car-line', color: '#00d4ff' },
            { label: 'Bu Hafta', value: kpiStats.thisWeek, icon: 'ri-calendar-line', color: '#00ff88' },
            { label: 'Bu Ay', value: kpiStats.thisMonth, icon: 'ri-bar-chart-2-line', color: '#ffd700' },
            { label: 'Pik Saat', value: kpiStats.peakHour, icon: 'ri-time-line', color: '#a855f7' },
          ].map(s => (
            <div key={s.label} className={`${t.cardBg} rounded-xl p-4 border ${t.border} transition-colors duration-300`}>
              <div className="w-9 h-9 flex items-center justify-center rounded-lg mb-3" style={{ backgroundColor: `${s.color}20` }}>
                <i className={`${s.icon} text-base`} style={{ color: s.color }}></i>
              </div>
              <div className={`${t.textPrimary} font-bold text-xl font-mono`}>{s.value}</div>
              <div className={`${t.textSecondary} text-xs mt-1`}>{s.label}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-4">
          
          {/* Saatlik Araç Sayımı Grafiği */}
          <div className={`col-span-2 ${t.cardBg} rounded-xl border ${t.border} p-4 transition-colors duration-300`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className={`${t.textPrimary} font-semibold text-sm`}>Saatlik Araç Sayımı</h3>
              <div className="flex gap-2">
                {roadKeys.map(r => (
                  <button
                    key={r.key}
                    onClick={() => setActiveRoad(r.key)}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-all cursor-pointer whitespace-nowrap`}
                    style={activeRoad === r.key ? { backgroundColor: r.color, color: '#000' } : { color: t.isDark ? '#8892a4' : '#6b7280', backgroundColor: t.isDark ? '#0f1117' : '#f3f4f6' }}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-end gap-1 h-40">
              {hourlyData.map((d) => {
                // val değerinin sıfır veya eksik olma durumuna karşı güvenlik önlemi
                const val = (d[activeRoad as keyof typeof d] as number) || 0; 
                const h = maxHourly > 0 ? (val / maxHourly) * 100 : 0;
                const activeColor = roadKeys.find(r => r.key === activeRoad)?.color || '#00d4ff';
                return (
                  <div key={d.hour} className="flex-1 flex flex-col items-center gap-1 group">
                    <div className="relative w-full flex items-end justify-center" style={{ height: '130px' }}>
                      <div className="w-full rounded-t transition-all duration-300 group-hover:opacity-80" style={{ height: `${h}%`, backgroundColor: activeColor, opacity: 0.7 }}></div>
                      <div className={`absolute -top-6 left-1/2 -translate-x-1/2 ${t.cardBg} ${t.textPrimary} text-[9px] px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 border ${t.border}`}>
                        {val}
                      </div>
                    </div>
                    <span className={`${t.textSecondary} text-[8px] rotate-45 origin-left`}>{d.hour.split(':')[0]}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Günlük Karşılaştırma */}
          <div className={`${t.cardBg} rounded-xl border ${t.border} p-4 transition-colors duration-300`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm mb-4`}>Günlük Karşılaştırma</h3>
            <div className="space-y-3">
              {dailyData.map(d => (
                <div key={d.day} className="flex items-center gap-3">
                  <span className={`${t.textSecondary} text-xs w-8`}>{d.day}</span>
                  <div className={`flex-1 ${t.isDark ? 'bg-[#0f1117]' : 'bg-[#f3f4f6]'} rounded-full h-2`}>
                    <div className="h-2 rounded-full bg-[#00d4ff] transition-all duration-500" style={{ width: maxDaily > 0 ? `${(d.count / maxDaily) * 100}%` : '0%' }}></div>
                  </div>
                  <span className={`${t.textPrimary} text-xs font-mono w-14 text-right`}>{d.count.toLocaleString('tr-TR')}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Yol Bazlı Araç Yoğunluğu */}
        <div className={`${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
          <div className={`px-4 py-3 border-b ${t.border}`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm`}>Yol Bazlı Araç Yoğunluğu</h3>
          </div>
          <div className={`divide-y ${t.divider}`}>
            {roadData.map(r => {
              const cfg = densityConfig[r.status as keyof typeof densityConfig];
              return (
                <div key={r.road} className={`flex items-center gap-4 px-4 py-3 ${t.hoverBg} transition-colors`}>
                  <div className="flex-1">
                    <div className={`${t.textPrimary} text-sm font-medium`}>{r.road}</div>
                    <div className={`${t.textSecondary} text-xs`}>{r.length}</div>
                  </div>
                  <div className="w-32">
                    <div className="flex justify-between mb-1">
                      <span className={`${t.textSecondary} text-xs`}>Yoğunluk</span>
                      <span className={`text-xs font-bold ${cfg.color}`}>{r.density}%</span>
                    </div>
                    <div className={`${t.isDark ? 'bg-[#0f1117]' : 'bg-[#f3f4f6]'} rounded-full h-1.5`}>
                      <div className={`h-1.5 rounded-full ${cfg.bar}`} style={{ width: `${r.density}%` }}></div>
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-semibold ${cfg.bg} ${cfg.color}`}>{cfg.label}</div>
                </div>
              );
            })}
          </div>
        </div>
        
      </div>
    </DashboardLayout>
  );
}