
import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle2, Shield, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

const Index = () => {
  const features = [
    {
      icon: <Shield className="h-6 w-6 text-cyberpurple" />,
      title: 'Advanced Threat Detection',
      description: 'Detect anomalies and security threats in real-time using state-of-the-art machine learning algorithms.'
    },
    {
      icon: <Zap className="h-6 w-6 text-cyberpurple" />,
      title: 'Federated Learning',
      description: 'Leverage distributed model training across your network without compromising data privacy.'
    },
    {
      icon: <CheckCircle2 className="h-6 w-6 text-cyberpurple" />,
      title: 'Explainable AI',
      description: 'Understand why anomalies were flagged with transparent, human-readable explanations.'
    }
  ];

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <header className="flex justify-end p-6">
        <div className="flex gap-4">
          <Button asChild variant="ghost">
            <Link to="/login">Log in</Link>
          </Button>
          <Button asChild>
            <Link to="/register">Sign up</Link>
          </Button>
        </div>
      </header>

      <main className="flex-grow">
        <section className="py-20 px-4 text-center bg-gradient-to-b from-background via-background to-secondary/10">
          <div className="max-w-3xl mx-auto">
            <Shield className="h-20 w-20 text-cyberpurple mx-auto mb-6" />
            <h1 className="text-gradient text-5xl sm:text-6xl font-bold mb-6">
              CloudShield Sentinel AI
            </h1>
            <p className="text-xl text-foreground/80 mb-8">
              Next-generation cloud security platform powered by Federated Learning and Explainable AI.
              Detect anomalies, prevent breaches, and understand threats in real-time.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button asChild size="lg" className="text-md">
                <Link to="/dashboard">
                  Launch Dashboard <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="text-md">
                <Link to="/register">Sign up for free</Link>
              </Button>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12">Cutting-Edge Security Features</h2>
            <div className="grid md:grid-cols-3 gap-8">
              {features.map((feature, index) => (
                <Card key={index} className="bg-secondary/40">
                  <CardContent className="p-6 flex flex-col items-center text-center">
                    <div className="mb-4 p-3 rounded-full bg-cyberpurple/10">
                      {feature.icon}
                    </div>
                    <h3 className="text-xl font-medium mb-2">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4 bg-cyberpurple/5 border-t border-cyberpurple/10">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-6">Ready to secure your cloud infrastructure?</h2>
            <p className="text-lg text-muted-foreground mb-8">
              Start detecting anomalies and preventing security breaches today.
            </p>
            <Button asChild size="lg">
              <Link to="/register">Get Started Now</Link>
            </Button>
          </div>
        </section>
      </main>

      <footer className="py-8 px-4 border-t border-border">
        <div className="max-w-6xl mx-auto text-center text-sm text-muted-foreground">
          <p>© 2025 CloudShield Sentinel AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
