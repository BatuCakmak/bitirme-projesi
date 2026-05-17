import { useState } from 'react';
import DashboardLayout from '@/components/feature/DashboardLayout';
import { cameras } from '@/mocks/cameras';
import { useTheme } from '@/hooks/useTheme';
import { useThemeColors } from '@/hooks/useThemeColors';

export default function Settings() {
  const [notifications, setNotifications] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('30');
  const [saved, setSaved] = useState(false);
  const { isDark, toggleTheme } = useTheme();
  const t = useThemeColors();

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const Toggle = ({ val, onToggle }: { val: boolean; onToggle: () => void }) => (
    <button
      onClick={onToggle}
      className={`relative w-11 h-6 rounded-full transition-colors cursor-pointer`}
      style={{ backgroundColor: val ? t.accent : t.isDark ? '#2a2d3e' : '#d1d5db' }}
    >
      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${val ? 'translate-x-6' : 'translate-x-1'}`}></div>
    </button>
  );

  return (
    <DashboardLayout title="Ayarlar" subtitle="Sistem Ayarları ve Kamera Yönetimi">
      <div className="space-y-4 max-w-5xl">
        {/* General settings */}
        <div className={`${t.cardBg} rounded-xl border ${t.border} p-5 transition-colors duration-300`}>
          <h3 className={`${t.textPrimary} font-semibold text-sm mb-4 flex items-center gap-2`}>
            <i className="ri-settings-3-line" style={{ color: t.accent }}></i> Genel Ayarlar
          </h3>
          <div className="space-y-4">
            {/* Dark/Light mode */}
            <div className={`flex items-center justify-between py-2 border-b ${t.border}`}>
              <div>
                <div className={`${t.textPrimary} text-sm`}>Karanlık Mod</div>
                <div className={`${t.textSecondary} text-xs`}>Koyu tema kullan (şu an: {isDark ? 'Koyu' : 'Açık'})</div>
              </div>
              <Toggle val={isDark} onToggle={toggleTheme} />
            </div>

            <div className={`flex items-center justify-between py-2 border-b ${t.border}`}>
              <div>
                <div className={`${t.textPrimary} text-sm`}>Bildirimler</div>
                <div className={`${t.textSecondary} text-xs`}>Trafik uyarıları ve sistem bildirimleri</div>
              </div>
              <Toggle val={notifications} onToggle={() => setNotifications(!notifications)} />
            </div>

            <div className={`flex items-center justify-between py-2 border-b ${t.border}`}>
              <div>
                <div className={`${t.textPrimary} text-sm`}>Otomatik Yenileme</div>
                <div className={`${t.textSecondary} text-xs`}>Verileri otomatik olarak güncelle</div>
              </div>
              <Toggle val={autoRefresh} onToggle={() => setAutoRefresh(!autoRefresh)} />
            </div>

            <div className="flex items-center justify-between py-2">
              <div>
                <div className={`${t.textPrimary} text-sm`}>Yenileme Aralığı</div>
                <div className={`${t.textSecondary} text-xs`}>Saniye cinsinden veri güncelleme sıklığı</div>
              </div>
              <select
                value={refreshInterval}
                onChange={e => setRefreshInterval(e.target.value)}
                className={`${t.inputBg} border ${t.border} ${t.textPrimary} text-sm rounded-lg px-3 py-1.5 cursor-pointer`}
              >
                <option value="10">10 saniye</option>
                <option value="30">30 saniye</option>
                <option value="60">1 dakika</option>
                <option value="300">5 dakika</option>
              </select>
            </div>
          </div>
        </div>

        {/* Camera management */}
        <div className={`${t.cardBg} rounded-xl border ${t.border} overflow-hidden transition-colors duration-300`}>
          <div className={`px-5 py-4 border-b ${t.border} flex items-center justify-between`}>
            <h3 className={`${t.textPrimary} font-semibold text-sm flex items-center gap-2`}>
              <i className="ri-camera-line" style={{ color: t.accent }}></i> Kamera Yönetimi
            </h3>
            <span className={`${t.textSecondary} text-xs`}>{cameras.length} kamera</span>
          </div>
          <div className={`divide-y ${t.divider}`}>
            {cameras.map(cam => (
              <div key={cam.id} className={`flex items-center gap-4 px-5 py-3 ${t.hoverBg} transition-colors`}>
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${cam.status === 'active' ? 'bg-[#00ff88]' : cam.status === 'warning' ? 'bg-[#ffd700]' : 'bg-[#ff4757]'}`}></div>
                <div className="flex-1 min-w-0">
                  <div className={`${t.textPrimary} text-sm font-medium`}>{cam.name}</div>
                  <div className={`${t.textSecondary} text-xs`}>{cam.location} • {cam.id}</div>
                </div>
                <div className={`${t.textSecondary} text-xs w-24 text-right`}>{cam.vehicleCount} araç/sa</div>
                <div className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  cam.status === 'active' ? 'bg-[#00ff88]/20 text-[#00c97a]' :
                  cam.status === 'warning' ? 'bg-[#ffd700]/20 text-[#e6c200]' :
                  'bg-[#ff4757]/20 text-[#ff4757]'
                }`}>
                  {cam.status === 'active' ? 'Aktif' : cam.status === 'warning' ? 'Uyarı' : 'Çevrimdışı'}
                </div>
                <button className={`w-7 h-7 flex items-center justify-center rounded-lg ${t.inputBg} ${t.hoverBg} transition-colors cursor-pointer`}>
                  <i className={`ri-edit-line ${t.textSecondary} text-xs`}></i>
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Save button */}
        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className={`px-6 py-2.5 rounded-lg text-sm font-semibold transition-all cursor-pointer whitespace-nowrap ${saved ? 'bg-[#00ff88] text-black' : 'text-black hover:opacity-90'}`}
            style={!saved ? { backgroundColor: t.accent } : {}}
          >
            {saved ? <><i className="ri-check-line mr-2"></i>Kaydedildi!</> : 'Ayarları Kaydet'}
          </button>
        </div>
      </div>
    </DashboardLayout>
  );
}
