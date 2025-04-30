'use client';

import { Box, Button, Container, Typography, Paper, Grid } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InsightsIcon from '@mui/icons-material/Insights';

export default function Home() {
  return (
    <Box className="min-h-screen py-8">
      <Container maxWidth="lg">
        {/* Hero Section */}
        <Box className="py-16 text-center">
          <Typography 
            variant="h2" 
            component="h1" 
            className="font-bold mb-4"
            gutterBottom
          >
            Welcome to Partners Research
          </Typography>
          <Typography 
            variant="h5" 
            component="h2" 
            className="text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            Your comprehensive platform for market research and competitive analysis
          </Typography>
          <Button 
            variant="contained" 
            size="large" 
            color="primary"
            startIcon={<SearchIcon />}
            className="mr-4"
          >
            Start Researching
          </Button>
          <Button 
            variant="outlined" 
            size="large"
          >
            Learn More
          </Button>
        </Box>

        {/* Features Section */}
        <Box className="py-16">
          <Typography 
            variant="h4" 
            component="h2" 
            className="text-center mb-12"
            gutterBottom
          >
            Key Features
          </Typography>

          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <Paper elevation={3} className="p-6 h-full">
                <Box className="text-center">
                  <SearchIcon color="primary" className="text-5xl mb-4" />
                  <Typography variant="h6" gutterBottom>
                    Market Research
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Access comprehensive market data, trends, and insights to make informed decisions.
                  </Typography>
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper elevation={3} className="p-6 h-full">
                <Box className="text-center">
                  <TrendingUpIcon color="primary" className="text-5xl mb-4" />
                  <Typography variant="h6" gutterBottom>
                    Competitive Analysis
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Stay ahead with detailed analysis of your competitors and market position.
                  </Typography>
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper elevation={3} className="p-6 h-full">
                <Box className="text-center">
                  <InsightsIcon color="primary" className="text-5xl mb-4" />
                  <Typography variant="h6" gutterBottom>
                    Data Visualization
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Interactive charts and graphs to visualize complex market data easily.
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Box>

        {/* Call to Action */}
        <Box className="py-16 text-center">
          <Typography 
            variant="h4" 
            component="h2" 
            gutterBottom
          >
            Ready to get started?
          </Typography>
          <Typography 
            variant="body1" 
            className="mb-8 max-w-xl mx-auto"
          >
            Join thousands of businesses that use Partners Research to make data-driven decisions.
          </Typography>
          <Button 
            variant="contained" 
            size="large" 
            color="primary"
          >
            Get Started Now
          </Button>
        </Box>
      </Container>
    </Box>
  );
}
