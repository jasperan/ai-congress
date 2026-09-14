# Civic deliberation workspace

## Capture notes

Actual Svelte browser captures in dark and light mode with three synthetic model records. Model counts reflect the fixture selection. No deliberation, response quality, or live model availability is claimed.

Web captures use Chromium at 1440 × 1080 (desktop) and 390 × 1080 (mobile), with reduced motion enabled. Screenshots are real rendered interfaces, not image-generated UI mockups. Raster renderer examples retain their native dimensions.

## Verification — 2026-09-14

`cd frontend && npm run build` passed, and `node --test tests/socket.test.mjs` passed. The default-branch baseline did not build because its imported streaming helper was missing; this change restores that helper with a no-silent-replay regression test. Existing document-upload accessibility and unused-selector warnings remain. Browser checks cover selection, theme switching, responsive controls, and interrupted streams.

Only the existing visual surfaces were changed. The screenshots are not evidence of end-to-end service availability, accessibility certification, or production performance.
