// Multi-Fruit Varieties Dataset (English & Tamil)
// 8 Primary Agricultural & Commercial Fruits

export const ALL_FRUITS = [
  {
    id: 'mango',
    name: 'Mango',
    tamilName: 'மாம்பழம்',
    latinName: 'Mangifera indica',
    icon: '🥭',
    varieties: [
      { id: 'sindhoora', name: 'Sindhoora (Apple Mango)', tamilName: 'சிந்தூரா / ஆப்பிள் மாம்பழம்', origin: 'Tamil Nadu & South India (Red-Blushed)' },
      { id: 'alphonso', name: 'Alphonso', tamilName: 'அல்போன்சா', origin: 'Ratnagiri / Konkan & South India' },
      { id: 'neelum', name: 'Neelum', tamilName: 'நீலம்', origin: 'Tamil Nadu & South India' },
      { id: 'banganapalli', name: 'Banganapalli', tamilName: 'பங்கனபள்ளி', origin: 'Andhra Pradesh & Tamil Nadu' },
      { id: 'malgova', name: 'Malgova', tamilName: 'மல்கோவா', origin: 'Salem & Dharmapuri, Tamil Nadu' },
      { id: 'totapuri', name: 'Totapuri (Kilimookku)', tamilName: 'கிளிமூக்கு மாங்காய்', origin: 'Tamil Nadu & Karnataka' },
      { id: 'kesar', name: 'Kesar', tamilName: 'கேசர்', origin: 'Gujarat & Western India' }
    ]
  },
  {
    id: 'banana',
    name: 'Banana',
    tamilName: 'வாழைப்பழம்',
    latinName: 'Musa acuminata',
    icon: '🍌',
    varieties: [
      { id: 'sevazhai', name: 'Red Banana (Sevazhai)', tamilName: 'செவ்வாழை', origin: 'Kanyakumari & Southern Tamil Nadu' },
      { id: 'poovan', name: 'Poovan', tamilName: 'பூவன் வாழைப்பழம்', origin: 'Trichy & Central Tamil Nadu' },
      { id: 'nendran', name: 'Nendran', tamilName: 'நேந்திரன் வாழைப்பழம்', origin: 'Kanyakumari & Kerala' },
      { id: 'rasthali', name: 'Rasthali', tamilName: 'ரஸ்தாளி வாழைப்பழம்', origin: 'Tiruchirappalli & Tamil Nadu' },
      { id: 'grand_naine', name: 'Grand Naine (G9)', tamilName: 'கிராண்ட் நைன்', origin: 'Commercial Plantations' }
    ]
  },
  {
    id: 'papaya',
    name: 'Papaya',
    tamilName: 'பப்பாளி',
    latinName: 'Carica papaya',
    icon: '🍈',
    varieties: [
      { id: 'red_lady', name: 'Red Lady (Taiwan 786)', tamilName: 'ரெட் லேடி பப்பாளி', origin: 'South Indian Farms' },
      { id: 'co7', name: 'Coimbatore Co-7 / Co-8', tamilName: 'கோ-7 பப்பாளி', origin: 'TNAU Coimbatore, Tamil Nadu' },
      { id: 'washington', name: 'Washington Papaya', tamilName: 'வாஷிங்டன் பப்பாளி', origin: 'Commercial Orchards' }
    ]
  },
  {
    id: 'apple',
    name: 'Apple',
    tamilName: 'ஆப்பிள்',
    latinName: 'Malus domestica',
    icon: '🍎',
    varieties: [
      { id: 'royal_delicious', name: 'Royal Delicious', tamilName: 'ரொயல் டெலிஷியஸ்', origin: 'Himachal Pradesh & Kashmir' },
      { id: 'ambri', name: 'Kashmiri Ambri', tamilName: 'காஷ்மீரி அம்ரி', origin: 'Kashmir Valley' },
      { id: 'granny_smith', name: 'Granny Smith', tamilName: 'கிரானி ஸ்மித் (பச்சை)', origin: 'High Altitude Orchards' }
    ]
  },
  {
    id: 'orange',
    name: 'Orange',
    tamilName: 'ஆரஞ்சு',
    latinName: 'Citrus sinensis',
    icon: '🍊',
    varieties: [
      { id: 'nagpur', name: 'Nagpur Orange', tamilName: 'நாக்பூர் ஆரஞ்சு', origin: 'Maharashtra & Central India' },
      { id: 'sathgudi', name: 'Sathgudi Sweet Orange', tamilName: 'சாத்துக்குடி', origin: 'Andhra Pradesh & Tamil Nadu' },
      { id: 'coorg_mandarin', name: 'Coorg Mandarin', tamilName: 'கூர்க் மாண்டரின்', origin: 'Coorg & Western Ghats' },
      { id: 'kinnow', name: 'Kinnow Citrus', tamilName: 'கின்னோ', origin: 'Punjab & Northern Orchards' }
    ]
  },
  {
    id: 'guava',
    name: 'Guava',
    tamilName: 'கொய்யா',
    latinName: 'Psidium guajava',
    icon: '🍐',
    varieties: [
      { id: 'allahabad_safeda', name: 'Allahabad Safeda', tamilName: 'அலகாபாத் சபேடா', origin: 'Uttar Pradesh & All India' },
      { id: 'lucknow_49', name: 'Lucknow 49 / Sardar', tamilName: 'சர்தார் கொய்யா', origin: 'Maharashtra & Tamil Nadu' },
      { id: 'taiwan_pink', name: 'Taiwan Pink Guava', tamilName: 'தைவான் பிங்க் கொய்யா', origin: 'Tropical Commercial Farms' },
      { id: 'arka_kiran', name: 'Arka Kiran (Red Pulp)', tamilName: 'அர்கா கிரண்', origin: 'IIHR Bangalore' }
    ]
  },
  {
    id: 'pomegranate',
    name: 'Pomegranate',
    tamilName: 'மாதுளை',
    latinName: 'Punica granatum',
    icon: '🔴',
    varieties: [
      { id: 'bhagwa', name: 'Bhagwa (Kesar / Sindhoori)', tamilName: 'பக்வா மாதுளை', origin: 'Maharashtra & Tamil Nadu' },
      { id: 'ganesh', name: 'Ganesh', tamilName: 'கணேஷ் மாதுளை', origin: 'Central & South India' },
      { id: 'arakta', name: 'Arakta Heavy Red', tamilName: 'அரக்டா', origin: 'Commercial Orchards' },
      { id: 'ruby', name: 'Ruby Dark Crimson', tamilName: 'ரூபி மாதுளை', origin: 'Hybrid Cultivar' }
    ]
  },
  {
    id: 'watermelon',
    name: 'Watermelon',
    tamilName: 'தர்பூசணி',
    latinName: 'Citrullus lanatus',
    icon: '🍉',
    varieties: [
      { id: 'sugar_baby', name: 'Sugar Baby', tamilName: 'சுகர் பேபி', origin: 'All India Summer Farms' },
      { id: 'kiran', name: 'Kiran (Striped Oblong)', tamilName: 'கிரண் தர்பூசணி', origin: 'Tamil Nadu & Karnataka' },
      { id: 'black_diamond', name: 'Black Diamond', tamilName: 'பிளாக் டைமண்ட்', origin: 'Commercial Riverbeds' },
      { id: 'icebox', name: 'Icebox Personal Melon', tamilName: 'ஐஸ்பாக்ஸ்', origin: 'Modern Hydroponic Orchards' }
    ]
  }
];
