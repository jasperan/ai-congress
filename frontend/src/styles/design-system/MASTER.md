# AI Congress Design System

## Product Type
- **Type**: AI collaboration platform
- **Industry**: Artificial Intelligence / Machine Learning
- **Audience**: Developers, researchers, power users

## Style & Aesthetic

### Primary Style
**Clean Modern Professional**
- Professional, trustworthy, tech-forward
- Not overly playful or playful; balanced approach
- Methodical, deliberative (fitting the "congress" theme)

### Secondary Styles
- **Minimalist**: Clean surfaces, generous whitespace
- **Rounded**: 0.5rem-0.75rem border radius for cards
- **Consistent**: Unified spacing system (0.5rem, 1rem, 1.5rem, 2rem)

## Color Palette

### Light Mode
- **Primary**: `#0284c7` (blue-600) - Trust, technology
- **Secondary**: `#8b5cf6` (violet-500) - AI, creativity
- **Success**: `#10b981` (emerald-500) - Voting, agreement
- **Warning**: `#f59e0b` (amber-500) - Confidence, caution
- **Danger**: `#ef4444` (red-500) - Errors, conflicts
- **Background**: `#f9fafb` (gray-50)
- **Surface**: `#ffffff`
- **Text Primary**: `#111827` (gray-900)
- **Text Secondary**: `#6b7280` (gray-500)
- **Border**: `#e5e7eb` (gray-200)

### Dark Mode
- **Primary**: `#38bdf8` (sky-400)
- **Secondary**: `#a78bfa` (violet-400)
- **Success**: `#34d399` (emerald-400)
- **Warning**: `#fbbf24` (amber-400)
- **Danger**: `#f87171` (red-400)
- **Background**: `#0f172a` (slate-900)
- **Surface**: `#1e293b` (slate-800)
- **Text Primary**: `#f9fafb` (gray-50)
- **Text Secondary**: `#94a3b8` (gray-400)
- **Border**: `#334155` (slate-700)

## Typography

### Font Family
**Inter** (preferred over system fonts)
- Clean, modern, excellent readability
- Available via Google Fonts: `font-family: 'Inter', system-ui, sans-serif;`

### Font Sizes
- **Display**: 1.5rem (24px) - Hero titles
- **H1**: 1.25rem (20px) - Main section headers
- **H2**: 1rem (16px) - Subsection headers
- **Body**: 0.875rem (14px) - Default text
- **Small**: 0.75rem (12px) - Metadata, labels

### Line Heights
- **Body**: 1.5-1.75
- **Display**: 1.2
- **Small**: 1.4

### Letter Spacing
- **Body**: -0.02em (subtly tightened)
- **Display**: -0.03em (tightened for impact)

## Spacing System

### Scale (Rem)
- **0.5rem**: 8px - Small gaps, tight spacing
- **1rem**: 16px - Base spacing unit
- **1.5rem**: 24px - Comfortable spacing between sections
- **2rem**: 32px - Section spacing, large gaps
- **3rem**: 48px - Large section spacing

### Examples
- **Card padding**: 1rem-1.5rem
- **Gap between items**: 0.5rem-1rem
- **Margin between sections**: 1.5rem-2rem
- **Container padding**: 1rem-1.5rem

## Border Radius

- **Small**: 0.375rem (6px) - Tags, badges, small elements
- **Medium**: 0.5rem (8px) - Buttons, inputs, small cards
- **Large**: 0.75rem (12px) - Cards, panels, dialogs
- **Full**: 9999px (pill shape) - Pills, close buttons

## Shadows

- **Small**: `0 1px 2px 0 rgba(0, 0, 0, 0.05)`
- **Medium**: `0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)`
- **Large**: `0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)`
- **Floating**: `0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)`

## Component Tokens

### Buttons
- **Primary**: Primary color, white text, medium radius
- **Secondary**: Surface color, primary color border/text
- **Disabled**: Opacity 0.5, no cursor pointer

### Inputs
- **Border**: Border color, 2px width
- **Focus**: Primary color ring, 2px, outline: none
- **Placeholder**: Text secondary color, 0.75rem
- **Padding**: 0.625rem-0.875rem (10px-14px)

### Cards
- **Background**: Surface color
- **Border**: Border color, 1px solid
- **Shadow**: Medium shadow
- **Radius**: Large radius
- **Padding**: 1rem-1.5rem

### Badges
- **Small**: 0.5rem padding
- **Border Radius**: Small or full radius
- **Background**: Light tint of color, text darker

## Accessibility (CRITICAL)

### Color Contrast
- **Normal text**: Minimum 4.5:1
- **Large text**: Minimum 3:1
- **UI elements**: Minimum 3:1
- **Disabled text**: At least 3:1 against background

### Focus States
- **Visible ring**: 2px solid primary color
- **Ring offset**: 2px
- **Outline**: none
- **Visible on all interactive elements**: buttons, inputs, links

### ARIA Labels
- **Icon-only buttons**: aria-label required
- **Links**: aria-label if text not descriptive
- **Form labels**: for attribute, associated with input
- **Custom controls**: aria-label or descriptive text

### Touch Targets
- **Minimum size**: 44x44px touch targets
- **Spacing**: Minimum 0.25rem between touch targets
- **Large touch targets**: Prefer 48x48px+ for important actions

### Keyboard Navigation
- **Tab order**: Matches visual order
- **Skip links**: Visible skip to content
- **Escape key**: Closes dialogs/modals
- **Enter/Space**: Activates buttons/links

## Performance (HIGH)

### Image Optimization
- **Format**: WebP when supported
- **Lazy loading**: Add loading="lazy" to below-fold images
- **Responsive**: srcset for multiple sizes
- **Alt text**: Descriptive alt text for all images

### Reduce Motion
- **Check**: prefers-reduced-motion media query
- **Animations**: Respect reduced motion preference
- **Duration**: Use 150-300ms for micro-interactions
- **Transform/opacity**: Use these for animations (better performance)

### Code Splitting
- **Dynamic imports**: For heavy components
- **Route-based splitting**: For Next.js (if applicable)
- **Code splitting**: Separate feature modules

### Layout Performance
- **Content-visibility**: For long lists
- **Will-change**: Only for animated elements
- **Transform**: Use for animations (GPU accelerated)
- **Avoid**: Layout thrashing, forced reflows

## Animation Guidelines

### Micro-interactions
- **Duration**: 150-300ms
- **Easing**: ease-out for entrance, ease-in-out for feedback
- **Transform/opacity**: Use these properties (GPU accelerated)
- **Avoid**: width/height animations (causes reflow)

### Loading States
- **Skeleton screens**: Before content loads
- **Spinners**: For short operations (<2s)
- **Progress bars**: For longer operations (>2s)

### Scroll Animations
- **Transform**: TranslateY for content revealing
- **Opacity**: Fade in for smooth appearance
- **Delay**: Staggered delays for list items
- **Threshold**: Use IntersectionObserver for scroll triggers

## Responsive Design (HIGH)

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Mobile Optimizations
- **Touch targets**: 44x44px minimum
- **Touch-friendly spacing**: 0.5rem minimum between interactive elements
- **Readable font size**: Minimum 16px body text
- **Horizontal scroll**: Avoid unless necessary
- **Fixed elements**: Account for height in layout calculations

### Tablet Optimizations
- **Grid layouts**: 2-column fallback
- **Flexible components**: Swipe gestures where appropriate
- **Optimized spacing**: Adjust margins/padding

### Desktop Optimizations
- **Wide layouts**: 2-column or 3-column layouts
- **Hover states**: Improve usability
- **Performance**: Use virtualization for long lists

## Web Interface Guidelines

### Semantic HTML
- **Headings**: H1 → H2 hierarchy
- **Forms**: Label elements, proper input types
- **Navigation**: nav, main, article, section tags
- **Landmarks**: Use aria landmarks for screen readers

### Meta Tags
- **Viewport**: width=device-width, initial-scale=1
- **Description**: SEO meta description
- **Theme-color**: For mobile browsers

### Security
- **HTTPS**: Required for all pages
- **CSP**: Content Security Policy
- **XSS**: Sanitize all user content

### SEO
- **Title**: Descriptive, <60 characters
- **Meta description**: <160 characters
- **Semantic tags**: H1-H6 hierarchy

## Anti-Patterns to Avoid

### Visual Issues
- ❌ Overuse of emojis as icons
- ❌ Inconsistent border radius
- ❌ Poor contrast ratios
- ❌ Generic system fonts
- ❌ Unnecessary animations
- ❌ Poor spacing consistency
- ❌ Confusing hover states
- ❌ Hidden interactive elements

### Accessibility Issues
- ❌ Color-only indicators
- ❌ No focus states
- ❌ No ARIA labels on icon buttons
- ❌ Non-semantic HTML
- ❌ Keyboard navigation issues
- ❌ Missing alt text

### Performance Issues
- ❌ Large bundle sizes
- ❌ No code splitting
- ❌ Blocking render
- ❌ Unoptimized images
- ❌ No lazy loading

## Implementation Notes

### Tailwind Configuration
- Use semantic color names (primary, surface, text-secondary)
- Extend theme with custom spacing and border radius
- Configure dark mode via class strategy

### Component Design
- Use CSS variables for dynamic values
- Keep components small and focused
- Use composition over inheritance
- Maintain consistent API across components

### Theme Implementation
- Use CSS custom properties (variables)
- Dark mode via .dark class on html/body
- Automatic detection of system preference
- Manual toggle with localStorage persistence

---

_This is the master design system. Page-specific overrides can be added in `design-system/pages/` directory._
