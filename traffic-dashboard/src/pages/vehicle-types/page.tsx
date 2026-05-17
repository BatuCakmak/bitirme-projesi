import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/feature/DashboardLayout';
import { useThemeColors } from '@/hooks/useThemeColors';

// Sadece senin modelinin 4 sınıfına (Otomobil, Kamyon/Tır, Otobüs, Motosiklet) uygun başlangıç verileri
const initialHourlyByType = [
  { hour: '06', car: 180, truck: 42, bus: 28, moto: 12 },
  { hour: '07', car: 320, truck: 68, bus: 45, moto: 22 },
  { hour: '08', car: 410, truck: 72, bus: 52, moto: 18 },
  { hour: '09', car: 290, truck: 58, bus: 41, moto: 24 },
  { hour: '10', car: 240, truck: 51, bus: 36, moto: 28 },
  { hour: '11', car: 220, truck: 48, bus: 33, moto: 31 },
  { hour: '12', car: 280, truck: 55, bus: 39, moto: 26 },
  { hour: '13', car: 295, truck: 57, bus: 40, moto: 24 },
  { hour: '14', car: 260, truck: 52, bus: 37, moto: 29 },
  { hour: '15', car: 275, truck: 54, bus: 38, moto: 32 },
  { hour: '16', car: 350, truck: 65, bus: 47, moto: 21 },
  { hour: '17', car: 420, truck: 70, bus: 53, moto: 16 },
  { hour: '18', car: 395, truck: 67, bus: 50, moto: 14 },
];

const initialTypeDetails = [
  { type: 'Otomobil', icon: 'ri-car-line', count: 8432, avg_speed: 38, peak: '08:00', color: '#00d4ff', pct: 67 },
  { type: 'Kamyon/Tır', icon: 'ri-truck-line', count: 1876, avg_speed: 52, peak: '10:00', color: '#ff6b35', pct: 15 },
  { type: 'Otobüs', icon: 'ri-bus-line', count: 1234, avg_speed: 28, peak: '07:30', color: '#00ff88', pct: 10 },
  { type: 'Motosiklet', icon: 'ri-motorbike-line', count: 654, avg_speed: 45, peak: '09:00', color: '#a855f7', pct: 8 },
];

const initialDistributionData = [
  { type: 'Otomobil', count: 8432, percentage: 67, color: '#00d4ff' },
  { type: 'Kamyon/Tır', count: 1876, percentage: 15, color: '#ff6b35' },
  { type: 'Otobüs', count: 1234, percentage: 10, color: '#00ff88' },
  { type: 'Motosiklet', count: 654, percentage: 8, color: '#a855f7' }
];

export default function VehicleTypes() {
  const t = useThemeColors();

  // 1. Durum (State) Yönetimi
  const [totalVehicles, setTotalVehicles] = useState(12196);
  const [typeDetails, setTypeDetails] = useState(initialTypeDetails);
  const [distributionData, setDistributionData] = useState(initialDistributionData);
  const [hourlyData, setHourlyData] = useState(initialHourlyByType);

  // 2. Canlı Veri Çekme (API Entegrasyonu)
  useEffect(() => {
    const fetchVehicleTypes = async () => {
      try {
        const res = await fetch('http://localhost:8080/api/vehicles/stats');
        if (res.ok) {
          const data = await res.json();

          const total = data.toplamArac > 0 ? data.toplamArac : 1; 
          setTotalVehicles(data.toplamArac || 0);

          // Üstteki kartlar ve detay tablosu için veriyi güncelle
          setTypeDetails(prev => prev.map(td => {
            let newCount = td.count;
            if (td.type === 'Otomobil') newCount = data.araba || 0;
            if (td.type === 'Kamyon/Tır') newCount = data.kamyonTir || 0;
            if (td.type === 'Otobüs') newCount = data.otobus || 0;
            if (td.type === 'Motosiklet') newCount = data.motor || 0;

            return { ...td, count: newCount, pct: Math.round((newCount / total) * 100) || 0 };
          }));

          // Donut grafik için veriyi güncelle
          setDistributionData(prev => prev.map(vd => {
            let newCount = vd.count;
            if (vd.type === 'Otomobil') newCount = data.araba || 0;
            if (vd.type === 'Kamyon/Tır') newCount = data.kamyonTir || 0;
            if (vd.type === 'Otobüs') newCount = data.otobus || 0;
            if (vd.type === 'Motosiklet') newCount = data.motor || 0;

            return { ...vd, count: newCount, percentage: Math.round((newCount / total) * 100) || 0 };
          }));

          // Saatlik Dağılım API'sini çekme (Eğer Java'da yazdıysan)
          const resHourlyTypes = await fetch('http://localhost:8080/api/vehicles/hourly-types');
          if (resHourlyTypes.ok) {
             const hourlyTypesJson = await resHourlyTypes.json();
             setHourlyData(hourlyTypesJson);
          }
        }
      } catch (error) {
        console.error("Araç türü verileri çekilemedi:", error);
      }
    };

    fetchVehicleTypes();
    const interval = setInterval(fetchVehicleTypes, 3000); // 3 saniyede bir güncelle
    return () => clearInterval(interval);
  }, []);

  const maxStack = Math.max(...hourlyData.map(h => h.car + h.truck + h.bus + h.moto));

  return (
    <DashboardLayout title="Araç Türleri" subtitle="Araç Türü Sınıflandırması ve Analizi">
      <div className="space-y-4">
        
        {/* Type cards - grid-cols-6 yerine 4 yapıldı */}
        <div className="grid grid-cols-4 gap-4">
          {typeDetails.map(tv => (
            <div key={tv.type} className={`${t.cardBg} rounded-xl p-4 border ${t.border} transition-colors duration-300`}>
              <div className="w-9 h-9 flex items-center justify-center rounded-lg mb-3" style={{ backgroundColor: `${tv.color}20` }}>
                <i className={`${tv.icon} text-base`} style={{ color: tv.color }}></i>
              </div>
              <div className={`${t.textPrimary} font-bold text-lg font-mono`}>{tv.count.toLocaleString('tr-TR')}</div>
              <div className={`${t.textSecondary} text-xs mt-0.5`}>{tv.type}</div>
              <div className={`mt-2 ${t.isDark ? 'bg-[#0f1117]' : 'bg-[#f3f4f6]'} rounded-full h-1`}>
                <div className="h-1 rounded-full transition-all duration-500" style={{ width: `${tv.pct}%`, backgroundColor: tv.color }}></div>
              </div>
              <div className="text-[10px] mt-1" style={{ color: tv.color }}>{tv.pct}%</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-4">
          
          {/* Donut chart */}
          <div className={`${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm mb-4`}>Dağılım Oranı</h3>
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-40 h-40">
                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                  {(() => {
                    let offset = 0;
                    return distributionData.map(v => {
                      const dash = v.percentage;
                      const el = (
                        <circle key={v.type} cx="50" cy="50" r="35" fill="none" stroke={v.color} strokeWidth="18"
                          strokeDasharray={`${dash} ${100 - dash}`} strokeDashoffset={-offset} className="transition-all duration-500" />
                      );
                      offset += dash;
                      return el;
                    });
                  })()}
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className={`${t.textPrimary} font-bold text-xl`}>{totalVehicles.toLocaleString('tr-TR')}</div>
                  <div className={`${t.textSecondary} text-[10px]`}>Toplam</div>
                </div>
              </div>
            </div>
            <div className="space-y-2">
              {distributionData.map(v => (
                <div key={v.type} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: v.color }}></div>
                    <span className={`${t.textSecondary} text-xs`}>{v.type}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`${t.textPrimary} text-xs font-mono`}>{v.count.toLocaleString('tr-TR')}</span>
                    <span className={`${t.textSecondary} text-[10px]`}>{v.percentage}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Stacked bar chart */}
          <div className={`col-span-2 ${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm mb-4`}>Saatlik Araç Türü Dağılımı</h3>
            <div className="flex items-end gap-1.5 h-44">
              {hourlyData.map(h => {
                const total = h.car + h.truck + h.bus + h.moto;
                const totalH = maxStack > 0 ? (total / maxStack) * 160 : 0;
                return (
                  <div key={h.hour} className="flex-1 flex flex-col items-center gap-1">
                    <div className="w-full flex flex-col-reverse rounded-t overflow-hidden transition-all duration-500" style={{ height: `${totalH}px` }}>
                      <div style={{ height: `${total > 0 ? (h.car / total) * 100 : 0}%`, backgroundColor: '#00d4ff' }}></div>
                      <div style={{ height: `${total > 0 ? (h.truck / total) * 100 : 0}%`, backgroundColor: '#ff6b35' }}></div>
                      <div style={{ height: `${total > 0 ? (h.bus / total) * 100 : 0}%`, backgroundColor: '#00ff88' }}></div>
                      <div style={{ height: `${total > 0 ? (h.moto / total) * 100 : 0}%`, backgroundColor: '#a855f7' }}></div>
                    </div>
                    <span className={`${t.textSecondary} text-[9px]`}>{h.hour}</span>
                  </div>
                );
              })}
            </div>
            <div className="flex flex-wrap gap-4 mt-3">
              {[['#00d4ff','Otomobil'],['#ff6b35','Kamyon/Tır'],['#00ff88','Otobüs'],['#a855f7','Motosiklet']].map(([c,l]) => (
                <div key={l as string} className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c as string }}></div>
                  <span className={`${t.textSecondary} text-xs`}>{l}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Detail table */}
        <div className={`${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
          <div className={`px-4 py-3 border-b ${t.border}`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm`}>Araç Türü Detay Tablosu</h3>
          </div>
          <table className="w-full">
            <thead>
              <tr className={`border-b ${t.border}`}>
                {['Araç Türü','Toplam Sayı','Pik Saat','Oran'].map(h => (
                  <th key={h} className={`px-4 py-2 text-left ${t.textSecondary} text-xs font-medium`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className={`divide-y ${t.divider}`}>
              {typeDetails.map(tv => (
                <tr key={tv.type} className={`${t.hoverBg} transition-colors`}>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 flex items-center justify-center rounded-lg" style={{ backgroundColor: `${tv.color}20` }}>
                        <i className={`${tv.icon} text-xs`} style={{ color: tv.color }}></i>
                      </div>
                      <span className={`${t.textPrimary} text-sm`}>{tv.type}</span>
                    </div>
                  </td>
                  <td className={`px-4 py-3 ${t.textPrimary} text-sm font-mono`}>{tv.count.toLocaleString('tr-TR')}</td>
                  <td className={`px-4 py-3 ${t.textSecondary} text-sm`}>{tv.peak}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className={`flex-1 ${t.isDark ? 'bg-[#0f1117]' : 'bg-[#f3f4f6]'} rounded-full h-1.5 w-20`}>
                        <div className="h-1.5 rounded-full transition-all duration-500" style={{ width: `${tv.pct}%`, backgroundColor: tv.color }}></div>
                      </div>
                      <span className="text-xs font-semibold" style={{ color: tv.color }}>{tv.pct}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </DashboardLayout>
  );
}