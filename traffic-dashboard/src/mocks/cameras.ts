export const cameras = [
  { 
    id: 'CAM-001', 
    name: 'Bağcılar Kavşağı', 
    location: 'E-5 Karayolu / Bağcılar', 
    lat: 41.0392, 
    lng: 28.8553, 
    status: 'active', 
    density: 'high', 
    vehicleCount: 342, 
    avgSpeed: 18, 
    thumbnail: 'https://readdy.ai/api/search-image?query=busy%20city%20intersection%20traffic%20jam%20cars%20trucks%20buses%20urban%20road%20aerial%20view%20night%20dark%20cinematic&width=640&height=360&seq=cam001&orientation=landscape',
    // YENİ EKLENEN KISIM: Python Flask canlı yayın URL'i
    streamUrl: 'http://localhost:5000/video_feed/cam1' 
  },
  { 
    id: 'CAM-002', 
    name: 'Mecidiyeköy Meydanı', 
    location: 'TEM Otoyolu / Mecidiyeköy', 
    lat: 41.0677, 
    lng: 28.9947, 
    status: 'active', 
    density: 'medium', 
    vehicleCount: 198, 
    avgSpeed: 34, 
    thumbnail: 'https://readdy.ai/api/search-image?query=urban%20highway...',
    // YENİ EKLENEN: İkinci videonun akışı
    streamUrl: 'http://localhost:5000/video_feed/cam2' 
  },
  { id: 'CAM-003', name: 'Kadıköy Köprüsü', location: 'D-100 / Kadıköy', lat: 40.9923, lng: 29.0233, status: 'active', density: 'low', vehicleCount: 87, avgSpeed: 62, thumbnail: 'https://readdy.ai/api/search-image?query=city%20bridge%20road%20light%20traffic%20smooth%20flow%20cars%20highway%20clear%20road%20daytime&width=640&height=360&seq=cam003&orientation=landscape' },
  { id: 'CAM-004', name: 'Fatih Bulvarı', location: 'Vatan Caddesi / Fatih', lat: 41.0082, lng: 28.9396, status: 'active', density: 'high', vehicleCount: 415, avgSpeed: 12, thumbnail: 'https://readdy.ai/api/search-image?query=congested%20city%20boulevard%20heavy%20traffic%20standstill%20cars%20buses%20trucks%20urban%20street%20night&width=640&height=360&seq=cam004&orientation=landscape' },
  { id: 'CAM-005', name: 'Üsküdar Sahil', location: 'Sahil Yolu / Üsküdar', lat: 41.0231, lng: 29.0151, status: 'active', density: 'medium', vehicleCount: 156, avgSpeed: 41, thumbnail: 'https://readdy.ai/api/search-image?query=coastal%20road%20traffic%20moderate%20cars%20seaside%20urban%20road%20evening%20golden%20hour&width=640&height=360&seq=cam005&orientation=landscape' },
  { id: 'CAM-006', name: 'Levent Tüneli', location: 'TEM / Levent Girişi', lat: 41.0793, lng: 29.0122, status: 'warning', density: 'high', vehicleCount: 289, avgSpeed: 22, thumbnail: 'https://readdy.ai/api/search-image?query=tunnel%20entrance%20highway%20traffic%20cars%20headlights%20dark%20urban%20infrastructure%20road&width=640&height=360&seq=cam006&orientation=landscape' },
  { id: 'CAM-007', name: 'Atatürk Havalimanı Yolu', location: 'E-5 / Bakırköy', lat: 40.9769, lng: 28.8146, status: 'active', density: 'low', vehicleCount: 64, avgSpeed: 78, thumbnail: 'https://readdy.ai/api/search-image?query=airport%20road%20highway%20light%20traffic%20clear%20road%20cars%20fast%20moving%20daytime&width=640&height=360&seq=cam007&orientation=landscape' },
  { id: 'CAM-008', name: 'Boğaz Köprüsü Girişi', location: '15 Temmuz Şehitler Köprüsü', lat: 41.0456, lng: 29.0339, status: 'active', density: 'medium', vehicleCount: 223, avgSpeed: 45, thumbnail: 'https://readdy.ai/api/search-image?query=suspension%20bridge%20approach%20road%20traffic%20cars%20moderate%20flow%20urban%20bridge%20evening&width=640&height=360&seq=cam008&orientation=landscape' },
];

export const cameraStats = {
  total: 48,
  active: 44,
  warning: 3,
  offline: 1,
};