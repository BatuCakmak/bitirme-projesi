import { useState } from 'react';
import DashboardLayout from '@/components/feature/DashboardLayout';
import { alternativeRoutes } from '@/mocks/traffic';
import { useThemeColors } from '@/hooks/useThemeColors';

export default function AlternativeRoutes() {
  const [selected, setSelected] = useState<string | null>(null);
  const t = useThemeColors();

  return (
    <DashboardLayout title="Alternatif Rotalar" subtitle="Trafik Durumuna Göre Rota Önerileri">
      <div className="space-y-4">
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Aktif Öneri', value: '5', icon: 'ri-route-line', color: '#00d4ff' },
            { label: 'Ort. Tasarruf', value: '12 dk', icon: 'ri-time-line', color: '#00ff88' },
            { label: 'Yoğun Güzergah', value: '3', icon: 'ri-alarm-warning-line', color: '#ff4757' },
          ].map(s => (
            <div key={s.label} className={`${t.cardBg} rounded-xl p-4 border ${t.border} flex items-center gap-4 transition-colors duration-300`}>
              <div className="w-10 h-10 flex items-center justify-center rounded-xl" style={{ backgroundColor: `${s.color}20` }}>
                <i className={`${s.icon} text-lg`} style={{ color: s.color }}></i>
              </div>
              <div>
                <div className={`${t.textPrimary} font-bold text-xl`}>{s.value}</div>
                <div className={`${t.textSecondary} text-xs`}>{s.label}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-5 gap-4">
          {/* Map */}
          <div className={`col-span-3 ${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
            <div className={`px-4 py-3 border-b ${t.border}`}>
              <h3 className={`${t.textPrimary} font-semibold text-sm`}>Rota Haritası</h3>
            </div>
            <div className="relative" style={{ height: '420px' }}>
              <img
                src="https://readdy.ai/api/search-image?query=Istanbul%20city%20road%20map%20dark%20theme%20aerial%20view%20highways%20routes%20alternative%20paths%20colored%20lines%20urban%20infrastructure%20night&width=700&height=420&seq=routemap01&orientation=landscape"
                alt="Route Map"
                className={`w-full h-full object-cover object-top ${t.isDark ? 'opacity-50' : 'opacity-60'}`}
              />
              <div className={`absolute inset-0 ${t.isDark ? 'bg-[#0f1117]/40' : 'bg-white/20'}`}></div>
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 700 420" preserveAspectRatio="none">
                <path d="M 80 210 Q 200 180 350 210 Q 480 240 620 210" stroke="#ff4757" strokeWidth="4" fill="none" strokeOpacity="0.8" />
                <path d="M 80 140 Q 200 120 350 140 Q 480 160 620 140" stroke="#ffd700" strokeWidth="3" fill="none" strokeOpacity="0.7" />
                <path d="M 80 210 Q 150 280 280 300 Q 420 320 620 210" stroke="#00d4ff" strokeWidth="3" fill="none" strokeDasharray="10,5" strokeOpacity="0.9" />
                <path d="M 80 140 Q 160 200 300 220 Q 450 240 620 140" stroke="#00ff88" strokeWidth="2.5" fill="none" strokeDasharray="8,4" strokeOpacity="0.8" />
                {[[80,210],[350,210],[620,210],[80,140],[620,140]].map(([x,y],i) => (
                  <circle key={i} cx={x} cy={y} r="6" fill="#fff" fillOpacity="0.9" />
                ))}
              </svg>
              <div className="absolute bottom-4 left-4 flex flex-col gap-2">
                <div className="flex items-center gap-2 bg-black/60 rounded-lg px-3 py-1.5">
                  <div className="w-6 h-0.5 bg-[#ff4757]"></div>
                  <span className="text-white text-xs">Yoğun Ana Güzergah</span>
                </div>
                <div className="flex items-center gap-2 bg-black/60 rounded-lg px-3 py-1.5">
                  <div className="w-6 h-0.5 bg-[#00d4ff]"></div>
                  <span className="text-white text-xs">Önerilen Alternatif</span>
                </div>
              </div>
            </div>
          </div>

          {/* Route list */}
          <div className="col-span-2 space-y-3">
            {alternativeRoutes.map(r => (
              <div
                key={r.id}
                onClick={() => setSelected(selected === r.id ? null : r.id)}
                className={`${t.cardBg} rounded-xl border transition-all cursor-pointer ${selected === r.id ? 'border-[#00d4ff]' : `${t.border} hover:border-[#00d4ff]/40`}`}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className={`${t.textPrimary} text-sm font-semibold`}>{r.from}</span>
                        <i className={`ri-arrow-right-line ${t.textSecondary} text-xs`}></i>
                        <span className={`${t.textPrimary} text-sm font-semibold`}>{r.to}</span>
                      </div>
                      <div className={`${t.textSecondary} text-xs mt-0.5`}>{r.altRoute}</div>
                    </div>
                    {r.status === 'recommended' && (
                      <span className="bg-[#00ff88]/20 text-[#00c97a] text-[10px] font-bold px-2 py-0.5 rounded-full whitespace-nowrap">ÖNERİLEN</span>
                    )}
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">
                      <i className="ri-time-line text-[#00d4ff] text-xs"></i>
                      <span className={`${t.textPrimary} text-sm font-bold`}>{r.altDuration} dk</span>
                      <span className="text-[#00c97a] text-xs">(-{r.saving} dk)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <i className={`ri-map-pin-line ${t.textSecondary} text-xs`}></i>
                      <span className={`${t.textSecondary} text-xs`}>{r.altDistance}</span>
                    </div>
                  </div>
                  {selected === r.id && (
                    <div className={`mt-3 pt-3 border-t ${t.border} space-y-2`}>
                      <div className="flex justify-between text-xs">
                        <span className={t.textSecondary}>Ana Güzergah:</span>
                        <span className="text-[#ff4757]">{r.mainDuration} dk ({r.mainDistance})</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className={t.textSecondary}>Alternatif:</span>
                        <span className="text-[#00c97a]">{r.altDuration} dk ({r.altDistance})</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className={t.textSecondary}>Kazanç:</span>
                        <span className="text-[#00d4ff] font-bold">{r.saving} dakika</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
