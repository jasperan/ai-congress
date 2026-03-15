// tui/internal/app/models.go
package app

import (
	"fmt"
	"io"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jasperan/ai-congress/tui/internal/api"
	"github.com/jasperan/ai-congress/tui/internal/theme"
)

// --- Messages ---

type modelsLoadedMsg struct {
	models []api.ModelInfo
	err    error
}

// --- modelItem implements list.Item ---

type modelItem struct {
	model    api.ModelInfo
	selected bool
}

func (i modelItem) FilterValue() string { return i.model.Name }
func (i modelItem) Title() string       { return i.model.Name }
func (i modelItem) Description() string {
	backend := i.model.Backend
	if backend == "" {
		backend = "ollama"
	}
	return fmt.Sprintf("Weight: %.2f | Size: %s | %s", i.model.Weight, formatSize(i.model.Size), backend)
}

func formatSize(bytes int64) string {
	const gb = 1024 * 1024 * 1024
	const mb = 1024 * 1024
	if bytes >= gb {
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	}
	return fmt.Sprintf("%.0f MB", float64(bytes)/float64(mb))
}

// --- Custom delegate ---

type modelDelegate struct{}

func (d modelDelegate) Height() int                             { return 3 }
func (d modelDelegate) Spacing() int                            { return 1 }
func (d modelDelegate) Update(_ tea.Msg, _ *list.Model) tea.Cmd { return nil }

func (d modelDelegate) Render(w io.Writer, m list.Model, index int, listItem list.Item) {
	item, ok := listItem.(modelItem)
	if !ok {
		return
	}

	isSelected := index == m.Index()

	// Checkbox
	checkbox := "[ ]"
	if item.selected {
		checkbox = theme.StatusActive.Render("[✓]")
	}

	var nameStyle lipgloss.Style
	if isSelected {
		nameStyle = lipgloss.NewStyle().Bold(true).Foreground(theme.Cyan)
	} else {
		nameStyle = theme.Title
	}
	name := nameStyle.Render(item.model.Name)
	weight := theme.MutedText.Render(fmt.Sprintf(" (w:%.2f)", item.model.Weight))

	backend := item.model.Backend
	if backend == "" {
		backend = "ollama"
	}
	desc := theme.MutedText.Render(fmt.Sprintf("  %s  [%s]", formatSize(item.model.Size), backend))

	cursor := "  "
	if isSelected {
		cursor = theme.Subtitle.Render("> ")
	}

	fmt.Fprintf(w, "%s%s %s%s\n%s", cursor, checkbox, name, weight, desc)
}

// --- ModelsModel ---

type ModelsModel struct {
	client         *api.Client
	list           list.Model
	loaded         bool
	err            error
	selectedModels map[string]bool
}

func NewModelsModel(client *api.Client) ModelsModel {
	delegate := modelDelegate{}
	l := list.New([]list.Item{}, delegate, 0, 0)
	l.Title = "Available Models"
	l.SetShowStatusBar(true)
	l.SetFilteringEnabled(true)
	l.Styles.Title = theme.Title
	l.SetShowHelp(false)

	return ModelsModel{
		client:         client,
		list:           l,
		selectedModels: make(map[string]bool),
	}
}

func (m ModelsModel) Init() tea.Cmd {
	client := m.client
	return func() tea.Msg {
		models, err := client.ListModels()
		return modelsLoadedMsg{models: models, err: err}
	}
}

func (m ModelsModel) Update(msg tea.Msg) (ModelsModel, tea.Cmd) {
	switch msg := msg.(type) {
	case modelsLoadedMsg:
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		items := make([]list.Item, len(msg.models))
		for i, model := range msg.models {
			items[i] = modelItem{model: model}
		}
		m.list.SetItems(items)
		m.loaded = true
		return m, nil

	case tea.KeyMsg:
		if m.list.FilterState() == list.Filtering {
			break
		}
		switch msg.String() {
		case " ":
			// Toggle selection
			if item, ok := m.list.SelectedItem().(modelItem); ok {
				name := item.model.Name
				if m.selectedModels[name] {
					delete(m.selectedModels, name)
				} else {
					m.selectedModels[name] = true
				}
				// Update items to reflect selection state
				items := m.list.Items()
				for i, li := range items {
					if mi, ok := li.(modelItem); ok {
						mi.selected = m.selectedModels[mi.model.Name]
						items[i] = mi
					}
				}
				m.list.SetItems(items)
			}
			return m, nil

		case "enter":
			// Proceed to chat with selected models
			var selected []string
			for name := range m.selectedModels {
				selected = append(selected, name)
			}
			if len(selected) == 0 {
				// Select current item if none selected
				if item, ok := m.list.SelectedItem().(modelItem); ok {
					selected = []string{item.model.Name}
				}
			}
			if len(selected) > 0 {
				return m, func() tea.Msg {
					return SwitchScreenMsg{
						Screen: ScreenChat,
						Data:   selected,
					}
				}
			}

		case "h":
			return m, func() tea.Msg {
				return SwitchScreenMsg{Screen: ScreenHistory}
			}

		case "q", "esc":
			return m, func() tea.Msg {
				return SwitchScreenMsg{Screen: ScreenSplash}
			}
		}
	}

	var cmd tea.Cmd
	m.list, cmd = m.list.Update(msg)
	return m, cmd
}

func (m ModelsModel) View(width, height int) string {
	if m.err != nil {
		errView := theme.ErrorText.Render("Failed to load models: "+m.err.Error()) +
			"\n\n" + theme.KeyName.Render("q") + theme.KeyHint.Render(" back")
		return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, errView)
	}

	m.list.SetSize(width, height-2)

	selectedCount := len(m.selectedModels)
	selInfo := ""
	if selectedCount > 0 {
		selInfo = theme.Subtitle.Render(fmt.Sprintf("  %d selected", selectedCount))
	}

	hints := theme.KeyName.Render("Space") + theme.KeyHint.Render(" select") +
		"  " + theme.KeyName.Render("Enter") + theme.KeyHint.Render(" chat") +
		"  " + theme.KeyName.Render("/") + theme.KeyHint.Render(" filter") +
		"  " + theme.KeyName.Render("h") + theme.KeyHint.Render(" history") +
		"  " + theme.KeyName.Render("q") + theme.KeyHint.Render(" back") +
		selInfo

	return m.list.View() + "\n" + hints
}
