package tui

import (
	"sort"
	"strconv"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/jasperan/ai-congress/gotui/internal/api"
)

// MemberStatus is one council member's progress through a run.
type MemberStatus int

// Member states, in the order a member passes through them.
const (
	MemberPending MemberStatus = iota
	MemberGenerating
	MemberComplete
	MemberFailed
)

// Label renders a member state for the status column.
func (s MemberStatus) Label() string {
	switch s {
	case MemberPending:
		return "pending"
	case MemberGenerating:
		return "generating"
	case MemberComplete:
		return "complete"
	case MemberFailed:
		return "failed"
	default:
		return "unknown"
	}
}

// Member is one participant in the running council.
type Member struct {
	Name     string
	Status   MemberStatus
	Response string
	Detail   string
}

// Council is the live state of one run, folded from the /ws/chat frames.
//
// It exists so the progress, tally and transcript views all read from one
// place, and so the whole thing can be driven from a test without a service.
type Council struct {
	Question string
	Mode     string

	Members []Member

	// Rounds is the deliberation record: one entry per completed round.
	Rounds []api.Round

	// VoteBreakdown is the classic weighted tally: one entry per distinct
	// answer, keyed by the service's normalised response text.
	VoteBreakdown map[string]api.VoteGroup
	// Semantic is the semantic-voting result, when that mode was chosen.
	Semantic *api.SemanticVote

	FinalAnswer string
	Confidence  float64
	AgentsUsed  []string

	// Phase is the human-readable line the service sent with its start frame.
	Phase string
	// Error is set when the service reported a failure mid-run.
	Error string
	// Done reports whether the run reached a final answer.
	Done bool
}

// NewCouncil seeds a council with the members the user selected, so every
// participant is visible as "pending" before its first token arrives.
func NewCouncil(question, mode string, members []string) *Council {
	council := &Council{
		Question:      question,
		Mode:          mode,
		VoteBreakdown: map[string]api.VoteGroup{},
	}
	for _, name := range members {
		council.Members = append(council.Members, Member{Name: name, Status: MemberPending})
	}
	return council
}

// member returns the index of a member by name, creating it if the service
// reports a participant that was not in the request.
//
// A new member is appended rather than ignored: the service can name council
// members by agent role (deliberation builds agents from a triad), and an
// unseen name silently dropped would make the transcript lie about who spoke.
func (c *Council) memberIndex(name string) int {
	if name == "" {
		return -1
	}
	for i := range c.Members {
		if c.Members[i].Name == name {
			return i
		}
	}
	c.Members = append(c.Members, Member{Name: name, Status: MemberPending})
	return len(c.Members) - 1
}

// Apply folds one stream frame into the council.
func (c *Council) Apply(event api.StreamEvent) {
	switch event.Type {
	case "start":
		c.Phase = event.Message

	case "status_init":
		for _, personality := range event.Personalities {
			index := c.memberIndex(personality.Name)
			if index >= 0 {
				c.Members[index].Detail = personality.Status
			}
		}

	case "status_update":
		index := c.memberIndex(event.Name)
		if index < 0 {
			return
		}
		switch {
		case strings.Contains(strings.ToLower(event.Status), "generat"):
			c.Members[index].Status = MemberGenerating
		case strings.Contains(strings.ToLower(event.Status), "complete"):
			c.Members[index].Status = MemberComplete
		}
		if event.Response != "" {
			c.Members[index].Response = event.Response
		}
		c.Members[index].Detail = event.Status

	case "chunk":
		index := c.memberIndex(event.Name)
		if index < 0 {
			return
		}
		// Live tokens: append so the transcript grows as the model writes.
		c.Members[index].Status = MemberGenerating
		c.Members[index].Response += event.Content

	case "model_response":
		name := event.Model
		if name == "" {
			name = event.Name
		}
		index := c.memberIndex(name)
		if index < 0 {
			return
		}
		c.Members[index].Status = MemberComplete
		c.Members[index].Response = event.Content

	case "final_answer":
		c.Done = true
		c.FinalAnswer = event.Content
		c.Confidence = event.Confidence
		if event.Mode != "" {
			c.Mode = event.Mode
		}
		if event.VoteBreakdown != nil {
			c.VoteBreakdown = event.VoteBreakdown
		}
		if event.SemanticVote != nil {
			c.Semantic = event.SemanticVote
		}
		if event.Data != nil {
			c.Rounds = event.Data.Rounds
			c.AgentsUsed = event.Data.AgentsUsed
			for _, position := range event.Data.FinalPositions {
				name := position.Model
				if position.Agent != "" {
					name = position.Agent
				}
				index := c.memberIndex(name)
				if index < 0 {
					continue
				}
				if position.Response != "" {
					c.Members[index].Response = position.Response
				}
				if position.Success {
					c.Members[index].Status = MemberComplete
				}
			}
		}

	case "error":
		c.Error = event.Message
	}

	// Any member still marked pending when the run has ended never reported:
	// say so rather than leaving a spinner that will never resolve.
	if c.Done {
		for i := range c.Members {
			if c.Members[i].Status == MemberPending {
				c.Members[i].Status = MemberFailed
			}
		}
	}
}

// Completed counts the members that produced a response.
func (c *Council) Completed() int {
	if c == nil {
		return 0
	}
	done := 0
	for _, member := range c.Members {
		if member.Status == MemberComplete {
			done++
		}
	}
	return done
}

// Running reports whether the council is still working.
func (c *Council) Running() bool {
	if c == nil {
		return false
	}
	return !c.Done && c.Error == ""
}

// ProgressFraction is the share of members that have finished, for a bar.
func (c *Council) ProgressFraction() float64 {
	if c == nil || len(c.Members) == 0 {
		return 0
	}
	return float64(c.Completed()) / float64(len(c.Members))
}

// RoundCount is the number of completed deliberation rounds.
func (c *Council) RoundCount() int {
	if c == nil {
		return 0
	}
	return len(c.Rounds)
}

// Tally is one row of a vote tally.
type Tally struct {
	Name   string
	Votes  float64
	Share  float64
	Winner bool
}

// Tallies ranks the vote groups, heaviest first.
//
// The service keys vote_breakdown by response text, so the rows are labelled by
// the models that pooled into each group -- that is what a reader wants to
// know -- and fall back to the answer text when the service named no model.
//
// Ties are broken by label so the order is stable: an unstable sort would make
// the tally jitter between renders.
func (c *Council) Tallies() []Tally {
	if c == nil || len(c.VoteBreakdown) == 0 {
		return nil
	}
	total := 0.0
	for _, group := range c.VoteBreakdown {
		total += group.Weight
	}
	rows := make([]Tally, 0, len(c.VoteBreakdown))
	for _, group := range c.VoteBreakdown {
		share := 0.0
		if total > 0 {
			share = group.Weight / total
		}
		rows = append(rows, Tally{Name: tallyLabel(group), Votes: group.Weight, Share: share})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Votes != rows[j].Votes {
			return rows[i].Votes > rows[j].Votes
		}
		return rows[i].Name < rows[j].Name
	})
	if len(rows) > 0 {
		rows[0].Winner = true
	}
	return rows
}

// tallyLabel names a vote group: the models that agreed, or the answer itself
// when the service reported no model names.
func tallyLabel(group api.VoteGroup) string {
	if models := strings.Join(group.Models, " + "); strings.TrimSpace(models) != "" {
		return models
	}
	if original := strings.TrimSpace(group.Original); original != "" {
		return oneLine(original)
	}
	return "unnamed answer"
}

// Winner is the model whose answer won, or "" when there were none.
func (c *Council) Winner() string {
	if c == nil {
		return ""
	}
	if c.Semantic != nil && c.Semantic.WinningModel != "" {
		return c.Semantic.WinningModel
	}
	rows := c.Tallies()
	if len(rows) == 0 {
		return ""
	}
	return rows[0].Name
}

// ConsensusShare is the winning share of the vote, in 0..1.
//
// It prefers the semantic result when present, because the semantic consensus
// is a judged agreement rather than a renormalised vote share, and reporting a
// 100% classic share (the only model that answered) as "unanimous" would
// overstate agreement.
func (c *Council) ConsensusShare() float64 {
	if c == nil {
		return 0
	}
	if c.Semantic != nil {
		return c.Semantic.Consensus
	}
	rows := c.Tallies()
	if len(rows) == 0 {
		return 0
	}
	return rows[0].Share
}

// --- rendering ---------------------------------------------------------------------

// TallyRows renders the vote tally as bars.
func TallyRows(council *Council, width int) string {
	rows := council.Tallies()
	if len(rows) == 0 {
		return Fog("No votes were counted.")
	}

	barWidth := contentWidth(width) / 4
	if barWidth < 6 {
		barWidth = 6
	}
	if barWidth > 24 {
		barWidth = 24
	}

	_, text, subtext, muted, _, success, _ := palette()
	var out strings.Builder
	for _, row := range rows {
		marker := "  "
		line := text
		if row.Winner {
			marker = "> "
			line = success
		}
		out.WriteString(marker)
		out.WriteString(line.Render(Pad(Truncate(row.Name, 22), 22)))
		out.WriteString(" ")
		out.WriteString(subtext.Render(Bar(row.Share, barWidth)))
		out.WriteString(" ")
		out.WriteString(muted.Render(Pad(Percent(row.Share), 5)))
		out.WriteString(" ")
		out.WriteString(muted.Render("(" + strconv.FormatFloat(row.Votes, 'f', 1, 64) + " votes)"))
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// MemberRows renders the live per-member progress column.
func MemberRows(council *Council, width int) string {
	if council == nil || len(council.Members) == 0 {
		return Fog("No council members yet.")
	}

	_, text, _, muted, _, _, _ := palette()
	var out strings.Builder
	for _, member := range council.Members {
		out.WriteString(text.Render(Pad(Truncate(member.Name, 24), 24)))
		out.WriteString(" ")
		out.WriteString(memberStatusStyle(member.Status).Render(Pad(member.Status.Label(), 10)))
		if member.Detail != "" && member.Status != MemberGenerating {
			out.WriteString(" ")
			out.WriteString(muted.Render(Truncate(member.Detail, contentWidth(width)/3)))
		}
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// memberStatusStyle colours a member state semantically.
func memberStatusStyle(status MemberStatus) lipgloss.Style {
	_, _, _, muted, danger, success, warning := palette()
	switch status {
	case MemberComplete:
		return success
	case MemberGenerating:
		return warning
	case MemberFailed:
		return danger
	default:
		return muted
	}
}

// RoundRows renders the deliberation record, round by round.
func RoundRows(council *Council, width int) string {
	if council == nil || len(council.Rounds) == 0 {
		return Fog("This run did not use the deliberation protocol.")
	}

	_, text, subtext, muted, _, _, _ := palette()
	var out strings.Builder
	for i, round := range council.Rounds {
		label := round.Name
		if label == "" {
			label = "round " + strconv.Itoa(i+1)
		}
		ok := 0
		for _, entry := range round.Outputs {
			if entry.Success {
				ok++
			}
		}
		out.WriteString(text.Render(label))
		out.WriteString("  ")
		out.WriteString(subtext.Render(strconv.Itoa(ok) + "/" + strconv.Itoa(len(round.Outputs)) + " positions"))
		out.WriteString("\n")
		for _, entry := range round.Outputs {
			speaker := entry.Agent
			if speaker == "" {
				speaker = entry.Model
			}
			marker := "  + "
			style := muted
			if !entry.Success {
				marker = "  ! "
			}
			out.WriteString(style.Render(marker + Truncate(speaker, 28) + "  " +
				Truncate(oneLine(entry.Response), contentWidth(width)/2)))
			out.WriteString("\n")
		}
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// TranscriptRows renders every council member's answer.
func TranscriptRows(council *Council, width int) string {
	if council == nil || len(council.Members) == 0 {
		return Fog("No responses yet.")
	}

	_, text, _, muted, danger, _, _ := palette()
	var out strings.Builder
	for _, member := range council.Members {
		out.WriteString(text.Bold(true).Render(Truncate(member.Name, 30)))
		out.WriteString(" ")
		out.WriteString(memberStatusStyle(member.Status).Render(member.Status.Label()))
		out.WriteString("\n")
		switch {
		case strings.TrimSpace(member.Response) != "":
			out.WriteString(muted.Render(wrapLines(member.Response, contentWidth(width))))
		case member.Status == MemberFailed:
			out.WriteString(danger.Render("no response"))
		default:
			out.WriteString(muted.Render("waiting..."))
		}
		out.WriteString("\n\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// VerdictRows renders the run's headline numbers.
func VerdictRows(council *Council, width int) string {
	if council == nil {
		return ""
	}
	var out strings.Builder
	out.WriteString(Stat("mode", council.Mode))
	out.WriteString("\n")
	out.WriteString(Stat("members", strconv.Itoa(len(council.Members))+
		" ("+strconv.Itoa(council.Completed())+" responded)"))
	out.WriteString("\n")
	out.WriteString(Stat("confidence", Percent(council.Confidence)))
	out.WriteString("\n")
	out.WriteString(Stat("consensus", Percent(council.ConsensusShare())))
	if winner := council.Winner(); winner != "" {
		out.WriteString("\n")
		out.WriteString(Stat("winner", winner))
	}
	if council.RoundCount() > 0 {
		out.WriteString("\n")
		out.WriteString(Stat("rounds", strconv.Itoa(council.RoundCount())))
	}
	if council.Semantic != nil && council.Semantic.DebateTriggered {
		out.WriteString("\n")
		out.WriteString(Stat("semantic debate", strconv.Itoa(council.Semantic.DebateRounds)+" rounds"))
	}
	out.WriteString("\n")
	out.WriteString(Stat("question", Truncate(oneLine(council.Question), contentWidth(width)/2)))
	return out.String()
}

// SemanticRows renders the semantic-voting clusters, when that mode ran.
func SemanticRows(council *Council, width int) string {
	if council == nil || council.Semantic == nil {
		return ""
	}
	semantic := council.Semantic
	_, text, subtext, muted, _, _, _ := palette()

	var out strings.Builder
	for _, cluster := range semantic.Clusters {
		label := cluster.Label
		if label == "" {
			label = "cluster " + strconv.Itoa(cluster.ID)
		}
		out.WriteString(text.Render(Truncate(label, contentWidth(width)/2)))
		out.WriteString("\n")
		out.WriteString(muted.Render("  " + strings.Join(cluster.Models, ", ")))
		out.WriteString("\n")
		for _, claim := range cluster.KeyClaims {
			out.WriteString(subtext.Render("  - " + Truncate(oneLine(claim), contentWidth(width)-4)))
			out.WriteString("\n")
		}
	}
	if semantic.DissentingSummary != "" {
		out.WriteString("\n")
		out.WriteString(Stat("dissent", Truncate(oneLine(semantic.DissentingSummary), contentWidth(width)-10)))
		out.WriteString("\n")
	}
	if out.Len() == 0 {
		return Fog("Semantic voting reported no clusters.")
	}
	return strings.TrimRight(out.String(), "\n")
}

// LeaderboardRows renders model standings.
func LeaderboardRows(rows []api.LeaderboardRow, width int) string {
	if len(rows) == 0 {
		return Fog("No runs have been recorded yet.")
	}
	_, text, _, muted, _, _, _ := palette()

	var out strings.Builder
	out.WriteString(muted.Render("  "+Pad("model", 28)+" "+Pad("weight", 8)+" "+Pad("win rate", 9)+"runs") + "\n")
	for _, row := range rows {
		out.WriteString(text.Render("  " + Pad(Truncate(row.Model, 28), 28) + " "))
		out.WriteString(text.Render(Pad(strconv.FormatFloat(row.Weight, 'f', 3, 64), 8) + " "))
		out.WriteString(text.Render(Pad(Percent(row.WinRate), 9)))
		out.WriteString(text.Render(strconv.FormatInt(row.Participations, 10)))
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// ModelRows renders the available models.
func ModelRows(models []api.ModelInfo, width int) string {
	if len(models) == 0 {
		return Fog("The service reported no models. Is Ollama running?")
	}
	_, text, _, muted, _, _, _ := palette()

	var out strings.Builder
	out.WriteString(muted.Render("  "+Pad("model", 32)+" "+Pad("weight", 8)+"backend") + "\n")
	for _, model := range models {
		out.WriteString(text.Render("  " + Pad(Truncate(model.Name, 32), 32) + " "))
		out.WriteString(text.Render(Pad(strconv.FormatFloat(model.Weight, 'f', 3, 64), 8) + " "))
		out.WriteString(text.Render(model.Backend))
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// TriadRows renders the deliberation triads.
func TriadRows(triads []api.TriadInfo, width int) string {
	if len(triads) == 0 {
		return Fog("No triads are configured.")
	}
	_, text, _, muted, _, _, _ := palette()

	var out strings.Builder
	for _, triad := range triads {
		out.WriteString(text.Bold(true).Render(Truncate(triad.Name, 30)))
		out.WriteString(" ")
		out.WriteString(muted.Render(Truncate(oneLine(triad.Description), contentWidth(width)/2)))
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// oneLine collapses whitespace so a multi-line model answer fits one row.
func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// wrapLines word-wraps text to width columns, collapsing existing whitespace.
func wrapLines(s string, width int) string {
	if width < 8 {
		width = 8
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return ""
	}
	var out strings.Builder
	line := 0
	for i, word := range words {
		length := len([]rune(word))
		if line > 0 && line+1+length > width {
			out.WriteString("\n")
			line = 0
		} else if i > 0 {
			out.WriteString(" ")
			line++
		}
		out.WriteString(word)
		line += length
	}
	return out.String()
}
