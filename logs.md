# 2026-07-21T15:54:46    conversation: 01ky2p7ewh3h9na19rz26qq6rh id: 01ky2p60pvr9kzecfdg4rwf272

Model: **gemini/gemini-3.6-flash** (resolved: **gemini-3.6-flash**)

## Prompt

Generate an SVG of a pelican riding a bicycle

## Response

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="100%">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGlow" cx="50%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#FFF9E6"/>
      <stop offset="60%" stop-color="#E0F2FE"/>
      <stop offset="100%" stop-color="#BAE6FD"/>
    </radialGradient>

    <!-- Beak & Pouch Gradients -->
    <linearGradient id="beakGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FCD34D"/>
      <stop offset="40%" stop-color="#F59E0B"/>
      <stop offset="100%" stop-color="#D97706"/>
    </linearGradient>

    <linearGradient id="pouchGrad" x1="0%" y1="0%" x2="30%" y2="100%">
      <stop offset="0%" stop-color="#FDBA74"/>
      <stop offset="50%" stop-color="#F97316"/>
      <stop offset="100%" stop-color="#C2410C"/>
    </linearGradient>

    <!-- Body Shading -->
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="70%" stop-color="#F1F5F9"/>
      <stop offset="100%" stop-color="#CBD5E1"/>
    </linearGradient>

    <!-- Bicycle Frame Gradient -->
    <linearGradient id="frameGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2DD4BF"/>
      <stop offset="50%" stop-color="#0D9488"/>
      <stop offset="100%" stop-color="#115E59"/>
    </linearGradient>

    <!-- Chrome / Metal Gradient -->
    <linearGradient id="metalGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#94A3B8"/>
      <stop offset="30%" stop-color="#F8FAFC"/>
      <stop offset="50%" stop-color="#CBD5E1"/>
      <stop offset="100%" stop-color="#64748B"/>
    </linearGradient>

    <!-- Wheel Rim Gradient -->
    <radialGradient id="rimGrad" cx="50%" cy="50%" r="50%">
      <stop offset="90%" stop-color="#E2E8F0"/>
      <stop offset="95%" stop-color="#94A3B8"/>
      <stop offset="100%" stop-color="#475569"/>
    </radialGradient>

    <!-- Shadow Filter -->
    <filter id="blurShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6"/>
    </filter>

    <!-- Reusable Spoke Def -->
    <g id="spokes">
      <line x1="0" y1="-60" x2="0" y2="60" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-60" y1="0" x2="60" y2="0" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-42" y1="-42" x2="42" y2="42" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-42" y1="42" x2="42" y2="-42" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-23" y1="-55" x2="23" y2="55" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-55" y1="-23" x2="55" y2="23" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-55" y1="23" x2="55" y2="-23" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
      <line x1="-23" y1="55" x2="23" y2="-55" stroke="#CBD5E1" stroke-width="1.5" opacity="0.7"/>
    </g>
  </defs>

  <!-- Background Layer -->
  <rect width="800" height="600" fill="url(#bgGlow)"/>

  <!-- Sun Backdrop Circle -->
  <circle cx="400" cy="260" r="200" fill="#FEF08A" opacity="0.5"/>

  <!-- Clouds / Motion Lines in Sky -->
  <path d="M 50 160 Q 80 150 110 160 T 170 160" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.8"/>
  <path d="M 680 120 Q 710 110 740 120" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.8"/>
  <path d="M 620 180 Q 650 170 680 180" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.6"/>

  <!-- Ground and Road -->
  <ellipse cx="400" cy="525" rx="360" ry="25" fill="#94A3B8" opacity="0.2"/>
  <line x1="40" y1="525" x2="760" y2="525" stroke="#0D9488" stroke-width="3" stroke-dasharray="12 12" opacity="0.4"/>

  <!-- Ground Shadow for Bicycle & Pelican -->
  <ellipse cx="230" cy="525" rx="75" ry="12" fill="#0F172A" filter="url(#blurShadow)" opacity="0.3"/>
  <ellipse cx="570" cy="525" rx="75" ry="12" fill="#0F172A" filter="url(#blurShadow)" opacity="0.3"/>
  <ellipse cx="400" cy="530" rx="180" ry="14" fill="#0F172A" filter="url(#blurShadow)" opacity="0.25"/>

  <!-- SPEED LINES (Behind) -->
  <path d="M 120 420 L 170 420 M 90 440 L 150 440 M 110 460 L 180 460" stroke="#38BDF8" stroke-width="3" stroke-linecap="round" opacity="0.5"/>

  <!-- === BICYCLE REAR WHEEL === -->
  <g id="rear-wheel">
    <!-- Outer Tire -->
    <circle cx="230" cy="440" r="75" fill="none" stroke="#1E293B" stroke-width="14"/>
    <!-- Rim -->
    <circle cx="230" cy="440" r="66" fill="none" stroke="url(#metalGrad)" stroke-width="4"/>
    <!-- Inner Shadow Rim -->
    <circle cx="230" cy="440" r="64" fill="none" stroke="#000000" opacity="0.15" stroke-width="1"/>
    <!-- Spokes -->
    <use href="#spokes" x="230" y="440"/>
    <!-- Hub -->
    <circle cx="230" cy="440" r="12" fill="url(#metalGrad)"/>
    <circle cx="230" cy="440" r="6" fill="#475569"/>
  </g>

  <!-- FAR LEG & PEDAL (Behind Frame) -->
  <g id="far-leg">
    <!-- Far Thigh & Calve -->
    <path d="M 360 310 L 320 370 L 335 415" fill="none" stroke="#E2E8F0" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M 360 310 L 320 370 L 335 415" fill="none" stroke="#CBD5E1" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Far Webbed Foot -->
    <path d="M 320 410 L 340 420 L 325 430 Z" fill="#F97316"/>
    <!-- Far Crank Arm -->
    <line x1="380" y1="440" x2="335" y2="415" stroke="#64748B" stroke-width="8" stroke-linecap="round"/>
    <!-- Far Pedal -->
    <rect x="320" y="410" width="25" height="10" rx="3" fill="#334155"/>
  </g>

  <!-- === BICYCLE FRAME (Back Parts) === -->
  <g id="bike-frame-back">
    <!-- Chainstay -->
    <line x1="230" y1="440" x2="380" y2="440" stroke="url(#frameGrad)" stroke-width="11" stroke-linecap="round"/>
    <!-- Seatstay -->
    <line x1="230" y1="440" x2="330" y2="300" stroke="url(#frameGrad)" stroke-width="9" stroke-linecap="round"/>
    <!-- Chain -->
    <path d="M 230 440 L 380 425 A 20 20 0 0 1 380 455 Z" fill="none" stroke="#475569" stroke-width="4" stroke-dasharray="6 3"/>
  </g>

  <!-- === PELICAN BODY & TAIL === -->
  <g id="pelican-body">
    <!-- Tail Feathers -->
    <path d="M 260 300 C 230 290 210 310 200 330 C 220 330 240 325 255 320 C 235 330 220 345 215 355 C 235 350 265 335 285 325 Z" fill="#E2E8F0"/>
    <path d="M 215 330 C 235 330 260 320 275 310" stroke="#CBD5E1" stroke-width="2" fill="none"/>

    <!-- Main Body Oval (Torso) -->
    <path d="M 270 310 C 260 250 330 220 390 240 C 440 255 430 320 390 360 C 340 390 280 360 270 310 Z" fill="url(#bodyGrad)"/>
    
    <!-- Body Contour / Feather Details -->
    <path d="M 310 350 C 340 370 380 360 400 330" fill="none" stroke="#CBD5E1" stroke-width="3" stroke-linecap="round"/>
    <path d="M 290 330 C 310 350 350 355 380 335" fill="none" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
  </g>

  <!-- === BICYCLE FRAME (Front Parts) & SADDLE === -->
  <g id="bike-frame-front">
    <!-- Bottom Bracket / Chainring -->
    <circle cx="380" cy="440" r="22" fill="url(#metalGrad)"/>
    <circle cx="380" cy="440" r="16" fill="#0D9488"/>

    <!-- Seat Tube -->
    <line x1="380" y1="440" x2="325" y2="290" stroke="url(#frameGrad)" stroke-width="12" stroke-linecap="round"/>
    <!-- Down Tube -->
    <line x1="380" y1="440" x2="510" y2="270" stroke="url(#frameGrad)" stroke-width="14" stroke-linecap="round"/>
    <!-- Top Tube -->
    <line x1="335" y1="310" x2="500" y2="270" stroke="url(#frameGrad)" stroke-width="11" stroke-linecap="round"/>

    <!-- Seat Post -->
    <line x1="328" y1="295" x2="320" y2="270" stroke="url(#metalGrad)" stroke-width="8"/>
    <!-- Saddle / Seat -->
    <path d="M 290 268 C 310 263 340 263 355 270 C 360 272 355 280 345 282 C 325 285 305 282 290 275 Z" fill="#1E293B"/>
    <path d="M 290 268 C 280 268 273 275 278 280 C 283 283 290 280 295 275 Z" fill="#0F172A"/>

    <!-- Head Tube -->
    <line x1="495" y1="250" x2="520" y2="300" stroke="url(#frameGrad)" stroke-width="14" stroke-linecap="round"/>
    <!-- Front Fork -->
    <line x1="515" y1="290" x2="570" y2="440" stroke="url(#frameGrad)" stroke-width="10" stroke-linecap="round"/>
  </g>

  <!-- === BICYCLE FRONT WHEEL === -->
  <g id="front-wheel">
    <!-- Outer Tire -->
    <circle cx="570" cy="440" r="75" fill="none" stroke="#1E293B" stroke-width="14"/>
    <!-- Rim -->
    <circle cx="570" cy="440" r="66" fill="none" stroke="url(#metalGrad)" stroke-width="4"/>
    <!-- Inner Shadow Rim -->
    <circle cx="570" cy="440" r="64" fill="none" stroke="#000000" opacity="0.15" stroke-width="1"/>
    <!-- Spokes -->
    <use href="#spokes" x="570" y="440"/>
    <!-- Hub -->
    <circle cx="570" cy="440" r="12" fill="url(#metalGrad)"/>
    <circle cx="570" cy="440" r="6" fill="#475569"/>
  </g>

  <!-- HANDLEBARS & STEM -->
  <g id="handlebars">
    <!-- Stem -->
    <path d="M 502 260 L 495 210 L 510 210" fill="none" stroke="url(#metalGrad)" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Handlebar Grip Base -->
    <path d="M 470 225 C 490 215 510 205 525 220" fill="none" stroke="url(#metalGrad)" stroke-width="7" stroke-linecap="round"/>
    <!-- Rubber Grips -->
    <path d="M 460 228 L 475 223" fill="none" stroke="#D97706" stroke-width="10" stroke-linecap="round"/>
    <!-- Classic Bicycle Bell -->
    <path d="M 512 206 A 8 8 0 0 1 524 200 L 520 212 Z" fill="#F59E0B"/>
    <circle cx="516" cy="204" r="6" fill="#FCD34D"/>
  </g>

  <!-- === PELICAN NECK, HEAD & BEAK === -->
  <g id="pelican-head-neck">
    <!-- Neck -->
    <path d="M 370 240 C 370 170 430 160 420 120" fill="none" stroke="url(#bodyGrad)" stroke-width="36" stroke-linecap="round"/>
    <path d="M 370 240 C 370 170 430 160 420 120" fill="none" stroke="#FFFFFF" stroke-width="28" stroke-linecap="round"/>
    <!-- Neck Contour Lines -->
    <path d="M 395 180 C 405 170 415 150 410 130" fill="none" stroke="#CBD5E1" stroke-width="2" opacity="0.6"/>

    <!-- Head Base -->
    <circle cx="425" cy="115" r="22" fill="#FFFFFF"/>

    <!-- FISH IN POUCH (Tail sticking out) -->
    <g id="fish-victim">
      <path d="M 510 145 C 525 135 540 130 550 135 C 545 145 535 150 520 152 Z" fill="#38BDF8"/>
      <!-- Fish Tail Fin -->
      <path d="M 545 133 L 565 125 C 563 135 563 140 568 148 Z" fill="#0284C7"/>
      <circle cx="520" cy="142" r="2" fill="#0F172A"/>
    </g>

    <!-- HUGE PELICAN BEAK & POUCH -->
    <!-- Elastic Pouch (Hanging Big) -->
    <path d="M 435 125 C 460 145 490 230 550 200 C 580 185 610 145 630 140 C 560 160 470 155 435 125 Z" fill="url(#pouchGrad)"/>
    <!-- Pouch Shadows/Folds -->
    <path d="M 470 160 C 490 190 520 210 550 200" fill="none" stroke="#C2410C" stroke-width="3" opacity="0.5"/>
    <path d="M 490 170 C 510 195 535 202 555 190" fill="none" stroke="#C2410C" stroke-width="2" opacity="0.4"/>

    <!-- Upper Mandible (Top Beak) -->
    <path d="M 425 105 C 470 105 560 125 640 138 C 645 139 642 143 635 144 C 550 140 470 130 425 122 Z" fill="url(#beakGrad)"/>
    <!-- Beak Hook Tip -->
    <path d="M 635 138 Q 643 140 640 146 Q 633 146 634 141 Z" fill="#B45309"/>

    <!-- Beak Center Line -->
    <path d="M 440 117 L 630 139" fill="none" stroke="#D97706" stroke-width="2" opacity="0.6"/>

    <!-- Eye Group -->
    <circle cx="426" cy="108" r="8" fill="#FFFFFF"/>
    <circle cx="428" cy="108" r="4.5" fill="#0F172A"/>
    <circle cx="430" cy="106" r="1.5" fill="#FFFFFF"/> <!-- Eye Highlight -->
    <!-- Brow/Expressive Line -->
    <path d="M 418 100 C 425 96 433 98 436 102" fill="none" stroke="#475569" stroke-width="2.5" stroke-linecap="round"/>

    <!-- CYCLING CAP -->
    <path d="M 408 104 C 405 85 425 75 440 88 C 445 98 442 106 442 106 Z" fill="#EF4444"/>
    <!-- Cap Brim / Visor -->
    <path d="M 432 94 C 445 92 460 96 468 102 C 455 104 442 102 432 94 Z" fill="#DC2626"/>
    <!-- Cap Stripe -->
    <path d="M 420 80 C 425 88 430 96 433 102" fill="none" stroke="#FFFFFF" stroke-width="4"/>
    <!-- Cap Button -->
    <circle cx="424" cy="80" r="3" fill="#FFFFFF"/>
  </g>

  <!-- NEAR LEG & PEDAL (In Front of Frame) -->
  <g id="near-leg">
    <!-- Hip Joint area -->
    <path d="M 370 340 L 415 410 L 395 465" fill="none" stroke="#F1F5F9" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M 370 340 L 415 410 L 395 465" fill="none" stroke="#CBD5E1" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>
    
    <!-- Feathered "Pants" / Thigh Base -->
    <path d="M 355 325 C 365 350 385 365 400 355 C 405 340 390 320 370 320 Z" fill="#E2E8F0"/>

    <!-- Near Webbed Foot wrapping Pedal -->
    <path d="M 380 460 C 390 455 415 460 425 468 C 415 478 395 480 380 472 Z" fill="#F97316"/>
    <path d="M 390 463 L 420 472 M 385 467 L 410 476" stroke="#EA580C" stroke-width="1.5"/>

    <!-- Near Crank Arm -->
    <line x1="380" y1="440" x2="395" y2="465" stroke="url(#metalGrad)" stroke-width="9" stroke-linecap="round"/>
    <circle cx="380" cy="440" r="6" fill="#475569"/>

    <!-- Near Pedal -->
    <rect x="385" y="462" width="30" height="8" rx="3" fill="#0F172A" transform="rotate(12 400 466)"/>
  </g>

  <!-- === PELICAN WING (Holding Handlebars) === -->
  <g id="pelican-wing">
    <!-- Main Wing Shape reaching forward -->
    <path d="M 330 260 C 350 240 410 250 465 230 C 450 250 420 275 380 285 C 350 292 335 280 330 260 Z" fill="#FFFFFF"/>
    
    <!-- Wing Outer / Layered Feathers -->
    <path d="M 350 268 C 380 255 430 242 475 228 C 460 248 440 265 410 278 Z" fill="#F8FAFC"/>
    
    <!-- Dark Primary Feathers (Wing Tips) wrapped on grip -->
    <path d="M 440 236 C 455 232 475 224 485 225 C 480 235 465 245 445 248 Z" fill="#1E293B"/>
    <path d="M 455 232 C 468 226 488 222 492 226 C 485 235 470 242 455 242 Z" fill="#334155"/>

    <!-- Wing Feather Textures -->
    <path d="M 360 265 C 390 260 420 250 445 240" fill="none" stroke="#CBD5E1" stroke-width="2"/>
    <path d="M 380 273 C 405 268 430 255 450 245" fill="none" stroke="#CBD5E1" stroke-width="2"/>
  </g>

  <!-- FOREGROUND / SPEED EFFECTS -->
  <!-- Wind / Motion Lines -->
  <path d="M 660 130 L 730 130 M 650 150 L 710 150 M 620 180 L 670 180" stroke="#38BDF8" stroke-width="3" stroke-linecap="round" opacity="0.6"/>
  <path d="M 450 80 C 470 75 490 75 510 80" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.7"/>

  <!-- Sweating / Effort Drops (Humor element) -->
  <path d="M 410 85 Q 400 80 403 75 Q 410 78 410 85 Z" fill="#38BDF8" opacity="0.8"/>
  <path d="M 395 95 Q 388 92 390 87 Q 396 89 395 95 Z" fill="#38BDF8" opacity="0.8"/>

  <!-- Wheels Spinning Motion Lines -->
  <path d="M 140 440 A 90 90 0 0 1 200 360" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.5"/>
  <path d="M 480 440 A 90 90 0 0 1 540 360" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.5"/>
</svg>
```

## Token usage

11 input, 10,770 output, {"candidatesTokenCount": 7249, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}], "thoughtsTokenCount": 3521, "serviceTier": "standard"}

# 2026-07-21T15:55:39    conversation: 01ky2p80q38herbxc0myc9fryr id: 01ky2p7mh9tjxa94xebcv090qk

Model: **gemini/gemini-3.5-flash-lite** (resolved: **gemini-3.5-flash-lite**)

## Prompt

Generate an SVG of a pelican riding a bicycle

## Response

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" width="100%" height="100%">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="skyGrad" cx="50%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#E0F7FA"/>
      <stop offset="100%" stop-color="#80DEEA"/>
    </radialGradient>

    <!-- Sun Gradient -->
    <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFF9C4" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#FFEE58" stop-opacity="0.2"/>
    </radialGradient>

    <!-- Pelican Body Gradient -->
    <linearGradient id="pelicanBody" x1="30%" y1="20%" x2="70%" y2="80%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="70%" stop-color="#ECEFF1"/>
      <stop offset="100%" stop-color="#CFD8DC"/>
    </linearGradient>

    <!-- Pouch Gradient -->
    <linearGradient id="pouchGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFCA28"/>
      <stop offset="50%" stop-color="#FFA000"/>
      <stop offset="100%" stop-color="#E65100"/>
    </linearGradient>

    <!-- Beak Gradient -->
    <linearGradient id="beakGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFEE58"/>
      <stop offset="100%" stop-color="#F57F17"/>
    </linearGradient>

    <!-- Wing Gradient -->
    <linearGradient id="wingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FAFAFA"/>
      <stop offset="60%" stop-color="#CFD8DC"/>
      <stop offset="100%" stop-color="#90A4AE"/>
    </linearGradient>

    <!-- Metal Frame Gradient -->
    <linearGradient id="metalFrame" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF5252"/>
      <stop offset="50%" stop-color="#D32F2F"/>
      <stop offset="100%" stop-color="#B71C1C"/>
    </linearGradient>

    <!-- Chrome/Silver Gradient -->
    <linearGradient id="chrome" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="50%" stop-color="#B0BEC5"/>
      <stop offset="100%" stop-color="#78909C"/>
    </linearGradient>

    <!-- Tire Gradient -->
    <linearGradient id="tireGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#424242"/>
      <stop offset="100%" stop-color="#212121"/>
    </linearGradient>

    <!-- Drop Shadow Filter -->
    <filter id="shadow" x="-10%" y="-10%" width="130%" height="130%">
      <feDropShadow dx="0" dy="10" stdDeviation="6" flood-color="#000000" flood-opacity="0.15"/>
    </filter>

    <filter id="bikeShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="5" dy="8" stdDeviation="4" flood-color="#000000" flood-opacity="0.2"/>
    </filter>

    <!-- Wheel Spoke Pattern -->
    <g id="spokes">
      <line x1="0" y1="0" x2="0" y2="130" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="91" y2="91" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="130" y2="0" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="91" y2="-91" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="0" y2="-130" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="-91" y2="-91" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="-130" y2="0" stroke="#B0BEC5" stroke-width="2.5"/>
      <line x1="0" y1="0" x2="-91" y2="91" stroke="#B0BEC5" stroke-width="2.5"/>
    </g>
  </defs>

  <!-- Background -->
  <rect width="800" height="800" fill="url(#skyGrad)"/>
  
  <!-- Decorative Sun -->
  <circle cx="650" cy="200" r="180" fill="url(#sunGrad)" />
  
  <!-- Ground / Road -->
  <path d="M0 680 C 200 660, 600 700, 800 670 L 800 800 L 0 800 Z" fill="#81C784"/>
  <path d="M0 690 C 200 670, 600 710, 800 680 L 800 800 L 0 800 Z" fill="#66BB6A"/>
  <!-- Road Surface -->
  <path d="M0 720 C 250 700, 550 740, 800 710 L 800 800 L 0 800 Z" fill="#78909C"/>
  <path d="M0 735 C 250 715, 550 755, 800 725" stroke="#CFD8DC" stroke-width="6" stroke-dasharray="25,20" fill="none"/>

  <!-- ================= BICYCLE ================= -->
  <g id="bicycle" filter="url(#bikeShadow)">
    <!-- Back Wheel -->
    <g transform="translate(250, 570)">
      <circle cx="0" cy="0" r="135" fill="none" stroke="#ECEFF1" stroke-width="8"/>
      <circle cx="0" cy="0" r="130" fill="none" stroke="url(#tireGrad)" stroke-width="16"/>
      <circle cx="0" cy="0" r="115" fill="none" stroke="#CFD8DC" stroke-width="3"/>
      <use href="#spokes"/>
      <circle cx="0" cy="0" r="16" fill="url(#chrome)"/>
      <circle cx="0" cy="0" r="6" fill="#37474F"/>
    </g>

    <!-- Front Wheel -->
    <g transform="translate(580, 570)">
      <circle cx="0" cy="0" r="135" fill="none" stroke="#ECEFF1" stroke-width="8"/>
      <circle cx="0" cy="0" r="130" fill="none" stroke="url(#tireGrad)" stroke-width="16"/>
      <circle cx="0" cy="0" r="115" fill="none" stroke="#CFD8DC" stroke-width="3"/>
      <use href="#spokes"/>
      <circle cx="0" cy="0" r="16" fill="url(#chrome)"/>
      <circle cx="0" cy="0" r="6" fill="#37474F"/>
    </g>

    <!-- Chain and Pedals (Center: 410, 570) -->
    <g transform="translate(410, 570)">
      <!-- Chain stay and seat stay links -->
      <path d="M 0 0 L -160 0" stroke="url(#metalFrame)" stroke-width="10" stroke-linecap="round"/>
      <circle cx="0" cy="0" r="32" fill="#37474F"/>
      <circle cx="0" cy="0" r="26" fill="url(#chrome)"/>
      <!-- Crank Arm & Pedal -->
      <path d="M 0 0 L 25 35" stroke="url(#chrome)" stroke-width="12" stroke-linecap="round"/>
      <rect x="15" y="30" width="30" height="12" rx="6" fill="#212121" transform="rotate(35 30 36)"/>
    </g>

    <!-- Bike Frame -->
    <!-- Chainstay: Back axle (250,570) to Bottom Bracket (410,570) -->
    <!-- Seat-tube: Bottom Bracket (410,570) to Seat (360, 390) -->
    <!-- Down-tube: Bottom Bracket (410,570) to Head-tube (530, 410) -->
    <!-- Seat-stay: Back axle (250,570) to Seat (360, 390) -->
    <!-- Top-tube: Seat (360, 390) to Head-tube (530, 410) -->
    <!-- Fork: Head-tube (530, 410) to Front Axle (580, 570) -->
    <path d="M 250 570 L 410 570 L 360 390 L 250 570" fill="none" stroke="url(#metalFrame)" stroke-width="14" stroke-linejoin="round" stroke-linecap="round"/>
    <path d="M 410 570 L 530 410 L 360 390" fill="none" stroke="url(#metalFrame)" stroke-width="14" stroke-linejoin="round" stroke-linecap="round"/>
    <path d="M 530 410 L 580 570" fill="none" stroke="url(#metalFrame)" stroke-width="14" stroke-linecap="round"/>

    <!-- Handlebars -->
    <path d="M 530 410 L 520 370 C 520 350, 540 330, 570 330 L 580 330" fill="none" stroke="url(#chrome)" stroke-width="12" stroke-linecap="round"/>
    <!-- Handlebar Grip -->
    <path d="M 570 325 C 585 325, 595 335, 595 350 C 595 360, 585 365, 570 365" fill="#212121" stroke="#212121" stroke-width="6" stroke-linecap="round"/>

    <!-- Saddle -->
    <path d="M 320 380 C 340 375, 380 375, 400 390 C 390 405, 330 405, 320 380 Z" fill="#212121" filter="url(#shadow)"/>
    <path d="M 330 382 C 350 378, 375 378, 390 387" fill="none" stroke="#424242" stroke-width="3"/>
  </g>

  <!-- ================= PELICAN ================= -->
  <g id="pelican" filter="url(#shadow)">
    
    <!-- Legs / Feet pedaling -->
    <!-- Back Leg (pedal up) -->
    <path d="M 390 520 C 400 480, 410 420, 395 390" fill="none" stroke="#FFA726" stroke-width="10" stroke-linecap="round"/>
    
    <!-- Front Leg (pedal down to crank 30,35 relative to 410,570 -> 435, 605) -->
    <path d="M 420 540 Q 440 580, 435 605" fill="none" stroke="#FB8C00" stroke-width="12" stroke-linecap="round"/>
    <!-- Webbed foot on pedal -->
    <path d="M 435 605 C 445 615, 460 610, 465 600 C 455 600, 440 600, 435 605 Z" fill="#F57C00"/>
    <path d="M 435 605 C 430 620, 445 625, 455 615" fill="none" stroke="#F57C00" stroke-width="4" stroke-linecap="round"/>

    <!-- Back Wing (Tucked/Active) -->
    <path d="M 370 420 C 340 430, 290 470, 310 520 C 330 510, 360 480, 390 450 Z" fill="#B0BEC5"/>

    <!-- Main Body -->
    <path d="M 330 360 C 420 340, 450 430, 410 520 C 370 600, 310 530, 300 460 C 290 390, 300 370, 330 360 Z" fill="url(#pelicanBody)"/>

    <!-- Fluffy chest feathers detail -->
    <path d="M 380 440 C 395 460, 390 490, 410 520 C 390 530, 370 510, 360 490" fill="#ECEFF1"/>
    <path d="M 365 475 C 380 490, 385 510, 395 530" fill="none" stroke="#CFD8DC" stroke-width="3" stroke-linecap="round"/>

    <!-- Neck & Head -->
    <!-- Elegant curved neck flowing into head -->
    <path d="M 340 370 C 330 310, 380 250, 430 250 C 460 250, 480 270, 470 290 C 460 310, 410 320, 390 360 Z" fill="url(#pelicanBody)"/>

    <!-- Cute Pelican Eye -->
    <circle cx="445" cy="275" r="9" fill="#212121"/>
    <circle cx="448" cy="272" r="3" fill="#FFFFFF"/> <!-- Eye highlight -->
    <!-- Eyebrow / Friendly curve -->
    <path d="M 432 263 Q 445 258, 458 266" fill="none" stroke="#78909C" stroke-width="3" stroke-linecap="round"/>
    <!-- Cute rosy cheek -->
    <ellipse cx="455" cy="288" rx="6" ry="4" fill="#FF8A80" opacity="0.6"/>

    <!-- Massive Beak Upper -->
    <path d="M 460 280 C 520 285, 590 300, 640 315 C 600 325, 520 315, 460 300 Z" fill="url(#beakGrad)"/>
    <!-- Beak Tip (Hooked) -->
    <path d="M 620 311 C 632 313, 642 315, 645 318 C 640 322, 630 320, 620 315 Z" fill="#E65100"/>

    <!-- Huge Pelican Pouch (The signature feature!) -->
    <path d="M 460 300 C 520 315, 600 325, 620 315 C 590 380, 500 400, 430 360 C 450 340, 455 320, 460 300 Z" fill="url(#pouchGrad)"/>
    <!-- Pouch pleats / texture lines -->
    <path d="M 520 340 Q 535 365, 550 380" fill="none" stroke="#FF8F00" stroke-width="2.5" opacity="0.6" stroke-linecap="round"/>
    <path d="M 490 330 Q 505 355, 515 375" fill="none" stroke="#FF8F00" stroke-width="2.5" opacity="0.6" stroke-linecap="round"/>
    <path d="M 550 345 Q 565 370, 580 375" fill="none" stroke="#FF8F00" stroke-width="2.5" opacity="0.6" stroke-linecap="round"/>

    <!-- Front Wing (Dynamic, resting on handlebar or waving) -->
    <path d="M 330 400 C 370 390, 480 340, 530 370 C 490 410, 400 480, 350 490 C 330 460, 320 420, 330 400 Z" fill="url(#wingGrad)"/>
    <!-- Wing Feather Details -->
    <path d="M 430 410 C 460 390, 490 375, 520 372" fill="none" stroke="#CFD8DC" stroke-width="3" stroke-linecap="round"/>
    <path d="M 410 435 C 440 415, 470 400, 500 395" fill="none" stroke="#CFD8DC" stroke-width="3" stroke-linecap="round"/>
    <path d="M 380 460 C 410 440, 440 425, 470 420" fill="none" stroke="#CFD8DC" stroke-width="3" stroke-linecap="round"/>

    <!-- Wing holding the handlebar -->
    <circle cx="570" cy="345" r="14" fill="url(#pelicanBody)"/>
    <path d="M 565 335 C 575 335, 585 345, 580 355 C 570 360, 560 350, 565 335 Z" fill="#CFD8DC"/>
  </g>

  <!-- Fun accessory: A little sailor hat on the pelican -->
  <g transform="translate(425, 230) rotate(-10)">
    <ellipse cx="25" cy="15" rx="30" ry="8" fill="#FFFFFF" stroke="#B0BEC5" stroke-width="2"/>
    <path d="M 10 15 L 12 -5 C 12 -12, 38 -12, 38 -5 L 40 15 Z" fill="#1976D2"/>
    <rect x="11" y="8" width="28" height="5" fill="#FFFFFF"/>
    <!-- Hat ribbon -->
    <path d="M 11 5 C 20 8, 30 8, 39 5" fill="none" stroke="#D32F2F" stroke-width="3"/>
  </g>

  <!-- Motion lines / Wind trails behind -->
  <g opacity="0.6">
    <path d="M 120 480 Q 180 480, 220 490" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
    <path d="M 80 540 Q 140 540, 180 550" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <path d="M 150 610 Q 190 610, 220 615" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
  </g>
</svg>
```

## Token usage

11 input, 5,534 output, {"candidatesTokenCount": 5534, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}], "serviceTier": "standard"}

