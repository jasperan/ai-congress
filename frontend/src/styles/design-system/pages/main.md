# AI Congress Main Page Overrides

## Page Specific Deviations

### Header/Navigation
- **Sticky positioning**: `sticky top-0`
- **Backdrop blur**: `backdrop-blur-md`
- **Transparency**: `bg-white/80` in light mode, `bg-gray-800/80` in dark mode
- **Border**: `border-gray-200` light, `border-gray-700` dark

### Main Content Area
- **Max width**: `max-w-7xl` (1280px)
- **Padding**: `px-4 sm:px-6 lg:px-8` (responsive)
- **Height calculation**: `h-[calc(100vh-4rem)]` for full-page layout

### Tab Navigation
- **Border**: `border-gray-200` light, `border-gray-700` dark
- **Active state**: `border-primary-500`, `text-primary-600` (or text-primary-400 in dark)
- **Inactive state**: `border-transparent`, `text-gray-500` (or text-gray-400 in dark)

### Modals/Sidebars
- **Backdrop**: `fixed inset-0 bg-black/50 backdrop-blur-sm`
- **Panel**: `fixed right-0 top-0 bottom-0 bg-white` (or `bg-gray-900` in dark)
- **Width**: `w-full md:w-2/3 lg:w-1/2`
- **Z-index**: `z-50`
- **Animation**: `animate-slide-up` for panel, `animate-fade-in` for backdrop

### Model Selector
- **Minimum touch targets**: 44x44px (rounded-lg with padding)
- **Spacing**: Gap of 0.5rem between model buttons
- **Selection state**: Solid background with checkmark
- **Hover state**: Subtle scale transform, improved visual feedback

### Message Areas
- **User messages**: Right-aligned, distinct background
- **Assistant messages**: Left-aligned, clean container
- **Spacing**: 1rem gap between messages
- **Animations**: `animate-slide-up` with staggered delays

### Loading States
- **Spinner**: Consistent across all components
- **Text**: Clear indication of what's happening
- **Duration**: Keep loading states as short as possible (<2s)

### Footer
- **Fixed positioning**: `fixed bottom-0 left-0 right-0`
- **Padding**: `py-2 px-4`
- **Transparency**: Solid background with subtle border
- **Content**: Centered, small text

---

This page-specific design system overrides the Master design system for the main application layout.
