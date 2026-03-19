EPL_TEAMS = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
    'Ipswich', 'Leicester', 'Liverpool', 'Man City', 'Man United',
    'Newcastle', 'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham',
    'Wolves'  # Common teams across seasons
]

TEAM_COLORS = {
    'Arsenal': '#EF0107',
    'Aston Villa': '#1A2C35', 
    'Bournemouth': '#000000',
    'Brentford': '#D50000',
    'Brighton': '#0057B8',
    'Burnley': '#6C4C2B',
    'Chelsea': '#024FA3',
    'Crystal Palace': '#1B4584',
    'Everton': '#003399',
    'Fulham': '#990000',
    'Liverpool': '#C8102E',
    'Man City': '#6CABDD',
    'Man United': '#DA291C',
    'Newcastle': '#000000',
    'Nottingham Forest': '#C8382D',
    'Tottenham': '#132257',
    'West Ham': '#4A2342',
    'Wolves': '#F41F37',
    # Defaults
    'Default': '#808080',
    'Ipswich': '#00A85A',
    'Leicester': '#004C9D',
    'Southampton': '#ED1A3B',
}

def get_team_color(team: str) -> str:
    """Get hex color for team."""
    return TEAM_COLORS.get(team.replace(' Manchester ', 'Man '), TEAM_COLORS['Default'])

